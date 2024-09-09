import os
import re

import yaml
from dotenv import load_dotenv
from jinja2 import Template
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, create_model

load_dotenv()


class PromptCaller:

    def _loadPrompt(self, file_path):
        with open(file_path, "r") as file:
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
        tag_pattern = r"<(system|user|assistant)>(.*?)</\1>"

        matches = re.findall(tag_pattern, body, re.DOTALL)

        for tag, content in matches:
            elements.append({"role": tag, "content": content.strip()})

        return elements

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
