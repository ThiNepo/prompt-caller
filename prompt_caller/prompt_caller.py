import os
import re

import requests
import yaml
from dotenv import load_dotenv
from jinja2 import Template
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from PIL import Image
from pydantic import BaseModel, Field, create_model

from io import BytesIO
import base64

load_dotenv()


class PromptCaller:

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

    def _parseJSXBody(self, body):
        elements = []
        tag_pattern = r"<(system|user|assistant|image)>(.*?)</\1>"

        matches = re.findall(tag_pattern, body, re.DOTALL)

        for tag, content in matches:
            elements.append({"role": tag, "content": content.strip()})

        return elements

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
            os.path.join("prompts", f"{promptName}.prompt")
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

                messages.append(
                    HumanMessage(
                        content=[
                            {
                                "type": "image_url",
                                "image_url": {"url": base64_image},
                            }
                        ]
                    )
                )

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

        chat = ChatOpenAI(**configuration)

        if output:
            dynamicModel = self.createPydanticModel(output)
            chat = chat.with_structured_output(dynamicModel)

        response = chat.invoke(messages)

        return response

    def agent(self, promptName, context=None, tools=None, allowed_steps=3):

        configuration, messages = self.loadPrompt(promptName, context)

        output = None

        if "output" in configuration:
            output = configuration.get("output")
            configuration.pop("output")

            for message in messages:
                if isinstance(message, SystemMessage):
                    message.content += "\nOnly use the tool DynamicModel when providing an output call."
                    break

        chat = ChatOpenAI(**configuration)

        # Register the tools
        if tools is None:
            tools = []

        # Transform functions in tools
        tools = [tool(t) for t in tools]

        tools_dict = {t.name.lower(): t for t in tools}

        if output:
            dynamicModel = self.createPydanticModel(output)

        tools.extend([dynamicModel])
        tools_dict["dynamicmodel"] = dynamicModel

        chat = chat.bind_tools(tools)

        try:
            # First LLM invocation
            response = chat.invoke(messages)
            messages.append(response)

            steps = 0
            while response.tool_calls and steps < allowed_steps:
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"].lower()

                    # If it's the final formatting tool, validate and return
                    if tool_name == "dynamicmodel":
                        return dynamicModel.model_validate(tool_call["args"])

                    selected_tool = tools_dict.get(tool_name)
                    if not selected_tool:
                        raise ValueError(f"Unknown tool: {tool_name}")

                    # Invoke the selected tool with provided arguments
                    tool_response = selected_tool.invoke(tool_call)
                    messages.append(tool_response)

                # If the latest message is a ToolMessage, re-invoke the LLM
                if isinstance(messages[-1], ToolMessage):
                    response = chat.invoke(messages)
                    messages.append(response)
                else:
                    break

                steps += 1

            # Final LLM call if the last message is still a ToolMessage
            if isinstance(messages[-1], ToolMessage):
                response = chat.invoke(messages)
                messages.append(response)

            return response

        except Exception as e:
            # Replace with appropriate logging in production
            raise RuntimeError("Error during agent process") from e
