import os
import re
import ast
from typing import Any, Optional, get_args, get_origin, Literal

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

    def _toModelNamePart(self, key):
        parts = re.split(r"[^A-Za-z0-9]+", key)
        normalized = [p.capitalize() for p in parts if p]
        return "".join(normalized) or "Field"

    def _validateFieldName(self, key, path):
        if "." in key:
            raise ValueError(
                f"Invalid output field '{path}.{key}': dotted keys are not supported. "
                "Use nested dictionaries instead (e.g. message: { gb: \"...\" })."
            )

        if key.count("?") > 1 or ("?" in key and not key.endswith("?")):
            raise ValueError(
                f"Invalid output field '{path}.{key}': optional marker '?' is only allowed at the end."
            )

    def _splitFieldName(self, key):
        is_optional = key.endswith("?")
        field_name = key[:-1] if is_optional else key
        if not field_name:
            raise ValueError("Output field name cannot be empty.")
        return field_name, is_optional

    def _splitTypeAndDescription(self, value):
        depth = 0
        for index, char in enumerate(value):
            if char == "[":
                depth += 1
            elif char == "]":
                depth = max(0, depth - 1)
            elif char == "|" and depth == 0:
                left = value[:index].strip()
                right = value[index + 1 :].strip()
                if left:
                    return left, right
        return None, value.strip()

    def _isTypeExpression(self, value):
        value = value.strip()
        if not value:
            return False

        primitive_types = {"string", "number", "integer", "boolean"}
        if value.lower() in primitive_types:
            return True
        if value.startswith("list[") and value.endswith("]"):
            return True
        if value.startswith("enum[") and value.endswith("]"):
            return True
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", value):
            return True

        return False

    def _mergeEnumDescriptions(self, type_expr, description, enum_descriptions, path):
        if not enum_descriptions:
            return description

        if not isinstance(enum_descriptions, dict):
            raise ValueError(
                f"Invalid output field '{path}': enum_descriptions must be a dictionary."
            )

        if not type_expr.startswith("enum["):
            raise ValueError(
                f"Invalid output field '{path}': enum_descriptions requires type enum[...]."
            )

        enum_inner = type_expr[5:-1].strip()
        enum_values = [v.strip() for v in enum_inner.split("|") if v.strip()]
        unknown_values = sorted(set(enum_descriptions.keys()) - set(enum_values))
        if unknown_values:
            raise ValueError(
                f"Invalid output field '{path}': enum_descriptions contains unknown values: "
                f"{', '.join(unknown_values)}."
            )

        parts = []
        if description:
            parts.append(description.strip())
        labeled_values = [
            f"{value}={enum_descriptions[value].strip()}"
            for value in enum_values
            if value in enum_descriptions and str(enum_descriptions[value]).strip()
        ]
        if labeled_values:
            parts.append("Enum meanings: " + "; ".join(labeled_values))
        return " ".join(parts).strip() or None

    def _resolveTypeExpression(
        self, type_expr, path, type_defs, named_types_cache
    ):
        expr = type_expr.strip()
        lower_expr = expr.lower()

        primitive_types = {
            "string": str,
            "number": float,
            "integer": int,
            "boolean": bool,
        }
        if lower_expr in primitive_types:
            return primitive_types[lower_expr]

        if expr.startswith("list[") and expr.endswith("]"):
            inner_expr = expr[5:-1].strip()
            if not inner_expr:
                raise ValueError(
                    f"Invalid output type '{type_expr}' at '{path}': list type requires inner type."
                )
            inner_type = self._resolveTypeExpression(
                inner_expr, f"{path}[]", type_defs, named_types_cache
            )
            return list[inner_type]

        if expr.startswith("enum[") and expr.endswith("]"):
            enum_inner = expr[5:-1].strip()
            enum_values = [v.strip() for v in enum_inner.split("|") if v.strip()]
            if not enum_values:
                raise ValueError(
                    f"Invalid output type '{type_expr}' at '{path}': enum requires at least one value."
                )
            return Literal.__getitem__(tuple(enum_values))

        if expr in named_types_cache:
            return named_types_cache[expr]

        if expr in type_defs:
            type_schema = type_defs[expr]
            if not isinstance(type_schema, dict):
                raise ValueError(
                    f"Invalid type definition 'types.{expr}': expected dictionary."
                )
            model_name = f"DynamicType{self._toModelNamePart(expr)}"
            named_types_cache[expr] = Any
            resolved_model = self._buildPydanticModel(
                model_name,
                type_schema,
                f"types.{expr}",
                type_defs=type_defs,
                named_types_cache=named_types_cache,
            )
            named_types_cache[expr] = resolved_model
            return resolved_model

        raise ValueError(
            f"Invalid output type '{type_expr}' at '{path}': unknown type expression."
        )

    def _buildField(self, model_name, key, value, path, type_defs, named_types_cache):
        self._validateFieldName(key, path)
        field_name, is_optional = self._splitFieldName(key)
        field_path = f"{path}.{field_name}"

        field_type = None
        description = None

        if isinstance(value, str):
            type_expr, parsed_description = self._splitTypeAndDescription(value)
            if type_expr is not None:
                field_type = self._resolveTypeExpression(
                    type_expr, field_path, type_defs, named_types_cache
                )
                description = parsed_description or None
            elif self._isTypeExpression(value):
                field_type = self._resolveTypeExpression(
                    value.strip(), field_path, type_defs, named_types_cache
                )
            else:
                field_type = str
                description = value
        elif isinstance(value, dict):
            if "type" in value:
                type_expr = value.get("type")
                if not isinstance(type_expr, str) or not type_expr.strip():
                    raise ValueError(
                        f"Invalid output field '{field_path}': 'type' must be a non-empty string."
                    )
                field_type = self._resolveTypeExpression(
                    type_expr, field_path, type_defs, named_types_cache
                )
                description = value.get("description")
                if description is not None and not isinstance(description, str):
                    raise ValueError(
                        f"Invalid output field '{field_path}': 'description' must be a string."
                    )
                enum_descriptions = value.get("enum_descriptions")
                description = self._mergeEnumDescriptions(
                    type_expr.strip(), description, enum_descriptions, field_path
                )
            else:
                nested_model_name = f"{model_name}{self._toModelNamePart(field_name)}"
                field_type = self._buildPydanticModel(
                    nested_model_name,
                    value,
                    field_path,
                    type_defs=type_defs,
                    named_types_cache=named_types_cache,
                )
        else:
            raise ValueError(
                f"Invalid output field '{field_path}': expected string or dictionary, "
                f"got {type(value).__name__}."
            )

        if is_optional:
            field_type = Optional[field_type]
            default_value = None
        else:
            default_value = ...

        if description is not None:
            field_info = Field(default_value, description=description)
        else:
            field_info = Field(default_value)

        return field_name, field_type, field_info

    def _buildPydanticModel(
        self,
        model_name,
        schema,
        path,
        type_defs=None,
        named_types_cache=None,
    ):
        if not isinstance(schema, dict):
            raise ValueError(f"Output schema at '{path}' must be a dictionary.")

        if type_defs is None:
            type_defs = {}
        if named_types_cache is None:
            named_types_cache = {}

        fields = {}

        for key, value in schema.items():
            (
                field_name,
                field_type,
                field_info,
            ) = self._buildField(
                model_name=model_name,
                key=key,
                value=value,
                path=path,
                type_defs=type_defs,
                named_types_cache=named_types_cache,
            )
            fields[field_name] = (field_type, field_info)

        return create_model(model_name, **fields)

    def createPydanticModel(self, dynamic_dict, type_defs=None):
        return self._buildPydanticModel(
            "DynamicModel",
            dynamic_dict,
            "output",
            type_defs=type_defs or {},
            named_types_cache={},
        )

    def call(self, promptName, context=None):
        configuration, messages = self.loadPrompt(promptName, context)

        output = configuration.pop("output", None)
        type_defs = configuration.pop("types", None)

        chat = self._createChat(configuration)

        if output:
            dynamicModel = self.createPydanticModel(output, type_defs=type_defs)
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

        prompt_output = configuration.pop("output", None)
        type_defs = configuration.pop("types", None)

        # Handle structured output from config
        dynamicOutput = None
        if output is None and prompt_output is not None:
            dynamicOutput = prompt_output

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
            response_format = self.createPydanticModel(
                dynamicOutput, type_defs=type_defs
            )

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
