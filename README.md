# PromptCaller

**PromptCaller** is a Python package for calling prompts in a specific format, using LangChain and the OpenAI API. It enables users to load prompts from a template, render them with contextual data, and make structured requests to the OpenAI API.

## Features

- **Load prompts** from a `.prompt` file containing a YAML configuration and a message template.
- **Invoke prompts** using LangChain and OpenAI API, with support for structured output.

## Installation

To install the package, simply run:

```bash
pip install prompt-caller
```

You will also need an `.env` file that contains your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

1. **Define a prompt file:**

Create a `.prompt` file in the `prompts` directory, e.g., `prompts/sample.prompt`:

```yaml
---
model: gpt-4o-mini
temperature: 0.7
max_tokens: 512
output:
  result: "Final result of the expression"
  explanation: "Explanation of the calculation"
---
<system>
You are a helpful assistant.
</system>

<user>
How much is {{expression}}?
</user>
```

This `.prompt` file contains:

- A YAML header for configuring the model and parameters.
- A template body using Jinja2 to inject the context (like `{{ expression }}`).
- Messages structured in a JSX-like format (`<system>`, `<user>`).

2. **Load and call a prompt:**

```python
from prompt_caller import PromptCaller

ai = PromptCaller()

response = ai.call("sample", {"expression": "3+8/9"})

print(response)
```

In this example:

- The `expression` value `3+8/9` is injected into the user message.
- The model will respond with both the result of the expression and an explanation, as specified in the `output` section of the prompt.

## How It Works

1. **\_loadPrompt:** Loads the prompt file, splits the YAML header from the body, and parses them.
2. **\_renderTemplate:** Uses the Jinja2 template engine to render the body with the provided context.
3. **\_parseJSXBody:** Parses the message body written in JSX-like tags to extract system and user messages.
4. **call:** Invokes the OpenAI API with the parsed configuration and messages, and handles structured output via dynamic Pydantic models.

## Build and Upload

To build the distribution and upload it to a package repository like PyPI, follow these steps:

1. **Build the distribution:**

   Run the following command to create both source (`sdist`) and wheel (`bdist_wheel`) distributions:

   ```bash
   python setup.py sdist bdist_wheel
   ```

   This will generate the distribution files in the `dist/` directory.

2. **Upload to PyPI using Twine:**

   Use `twine` to securely upload the distribution to PyPI:

   ```bash
   twine upload dist/*
   ```

   Ensure you have configured your PyPI credentials before running this command. You can find more information on configuring credentials in the [Twine documentation](https://twine.readthedocs.io/).

## License

This project is licensed under the **Apache License 2.0**. You may use, modify, and distribute this software as long as you provide proper attribution and include the full text of the license in any distributed copies or derivative works.
