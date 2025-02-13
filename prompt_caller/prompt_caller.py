import os
import re

import requests
import yaml
from dotenv import load_dotenv
from jinja2 import Template
from langchain_core.messages import HumanMessage, SystemMessage
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
