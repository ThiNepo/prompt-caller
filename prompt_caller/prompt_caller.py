import os
import re
import ast

import requests
import yaml
from dotenv import load_dotenv
from jinja2 import Template
from langgraph.types import Command
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
from pydantic import BaseModel, Field, create_model

from io import BytesIO
import base64

load_dotenv()


class PromptCaller:
    def __init__(self, promptPath="prompts"):
        self.promptPath = promptPath

    def _loadPrompt(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # Split YAML header and the body
        header, body = content.split("---", 2)[1:]

        # Parse the YAML header
        model_config = yaml.safe_load(header.strip())

        # Step 2: Parse the JSX body and return it
        return model_config, body.strip()

    def _renderTemplate(self, body, context):
        template = Template(body)
        return template.render(context)

    import re

    def _parseJSXBody(self, body):
        elements = []
        # 1. Regex to find tags, attributes string, and content
        tag_pattern = r"<(system|user|assistant|image)([^>]*)>(.*?)</\1>"

        # 2. Regex to find key="value" pairs within the attributes string
        attr_pattern = r'(\w+)\s*=\s*"(.*?)"'

        matches = re.findall(tag_pattern, body, re.DOTALL)

        for tag, attrs_string, content in matches:
            # 3. Parse the attributes string (e.g., ' tag="image 1"') into a dict
            attributes = {}
            if attrs_string:
                attr_matches = re.findall(attr_pattern, attrs_string)
                for key, value in attr_matches:
                    attributes[key] = value

            element = {"role": tag, "content": content.strip()}

            # 4. Add the attributes to our element dict if they exist
            if attributes:
                element["attributes"] = attributes

            elements.append(element)

        return elements

    def _createChat(self, configuration):
        if configuration.get("model") is not None and configuration.get(
            "model"
        ).startswith("gemini"):
            return ChatGoogleGenerativeAI(**configuration)
        else:
            return ChatOpenAI(**configuration)

    def getImageBase64(self, url: str) -> str:
        response = requests.get(url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_base64}"

    def loadPrompt(self, promptName, context=None):
        # initialize context
        if context is None:
            context = {}

        configuration, template = self._loadPrompt(
            os.path.join(self.promptPath, f"{promptName}.prompt")
        )

        template = self._renderTemplate(template, context)

        parsedMessages = self._parseJSXBody(template)

        messages = []

        for message in parsedMessages:
            if message.get("role") == "system":
                messages.append(SystemMessage(content=message.get("content")))

            if message.get("role") == "user":
                messages.append(HumanMessage(content=message.get("content")))

            if message.get("role") == "image":
                base64_image = message.get("content")

                if base64_image.startswith("http"):
                    base64_image = self.getImageBase64(base64_image)

                content = [
                    {
                        "type": "image_url",
                        "image_url": {"url": base64_image},
                    }
                ]

                tag = message.get("attributes", {}).get("tag")
                if tag:
                    content.append({"type": "text", "text": f"({tag})"})

                messages.append(HumanMessage(content=content))

        return configuration, messages

    def createPydanticModel(self, dynamic_dict):
        # Create a dynamic Pydantic model from the dictionary
        fields = {
            key: (str, Field(description=f"Description for {key}"))
            for key in dynamic_dict.keys()
        }
        # Dynamically create the Pydantic model with the fields
        return create_model("DynamicModel", **fields)

    def call(self, promptName, context=None):
        configuration, messages = self.loadPrompt(promptName, context)

        output = None

        if "output" in configuration:
            output = configuration.get("output")
            configuration.pop("output")

        chat = self._createChat(configuration)

        if output:
            dynamicModel = self.createPydanticModel(output)
            chat = chat.with_structured_output(dynamicModel)

        response = chat.invoke(messages)

        return response

    def _create_media_middleware(self):
        """Middleware to handle tool responses that contain media content (images, PDFs)."""

        @wrap_tool_call
        def handle_media_response(request, handler):
            result = handler(request)

            if hasattr(result, "content"):
                content = result.content

                if isinstance(content, str) and content.startswith("["):
                    try:
                        content = ast.literal_eval(content)
                    except (ValueError, SyntaxError):
                        pass

                # Check if content is media (image or PDF)
                if (
                    isinstance(content, list)
                    and content
                    and isinstance(content[0], dict)
                ):
                    is_media = "image_url" in content[0] or (  # Image
                        "input_file" == content[0].get("type", "").strip()
                    )  # PDF
                    if is_media:
                        filename = content[0].get("filename", "document.pdf")
                        tool_call_id = request.tool_call["id"]

                        tool_msg = ToolMessage(
                            tool_call_id=tool_call_id,
                            content=f"The file '{filename}' was loaded and is attached below for visual review.",
                        )

                        return Command(
                            update={
                                "messages": [tool_msg, HumanMessage(content=content)]
                            }
                        )

            return result

        return handle_media_response

    def agent(
        self, promptName, context=None, tools=None, output=None, allowed_steps=10
    ):
        configuration, messages = self.loadPrompt(promptName, context)

        # Handle structured output from config
        dynamicOutput = None
        if output is None and "output" in configuration:
            dynamicOutput = configuration.pop("output")

        chat = self._createChat(configuration)

        # Prepare tools
        if tools is None:
            tools = []
        tools = [tool(t) for t in tools]

        # Handle response format (structured output)
        response_format = None
        if output:
            response_format = output
        elif dynamicOutput:
            response_format = self.createPydanticModel(dynamicOutput)

        # Extract system message for create_agent
        system_prompt = None
        user_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_prompt = msg.content
            else:
                user_messages.append(msg)

        # Create and invoke agent
        agent_graph = create_agent(
            model=chat,
            tools=tools,
            system_prompt=system_prompt,
            response_format=response_format,
            middleware=[self._create_media_middleware()],
        )

        result = agent_graph.invoke(
            {"messages": user_messages}, {"recursion_limit": allowed_steps}
        )

        # Return structured output or last message
        if response_format and result.get("structured_response"):
            return result["structured_response"]
        return result["messages"][-1]
