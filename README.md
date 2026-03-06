![PyPI - Python Version](https://img.shields.io/pypi/pyversions/prompt-caller) ![PyPI](https://img.shields.io/pypi/v/prompt-caller)

[![PyPI - Downloads](https://img.shields.io/pypi/dm/prompt-caller)](https://pypi.org/project/prompt-caller/) [![Discord](https://img.shields.io/discord/479923444017004556?label=discord)](https://discord.gg/jWWDRyD5Nu)  [![YouTube Channel Subscribers](https://img.shields.io/youtube/channel/subscribers/UCMb90JgsFJpZyZzdmWCaCTg?style=social)](https://www.youtube.com/channel/UCMb90JgsFJpZyZzdmWCaCTg)

![Logo](images//logo.png)

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

## CLI Skill Installation

PromptCaller ships with a CLI command to install a PromptCaller skill pack into `.agents/skills`.

```bash
prompt-caller install
```

Alternative module invocation:

```bash
python -m prompt_caller install
```

By default, this installs to `.agents/skills/prompt-caller` and overwrites existing files.

## Usage

1. **Define a prompt file:**

Create a `.prompt` file in the `prompts` directory, e.g., `prompts/sample.prompt`:

```yaml
---
model: gpt-5.2
reasoning_effort: medium
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

- A YAML-like header for configuring the model and parameters.
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

### Advanced Prompt Example (`sample-5.2-complete.prompt`)

Use this when you want strongly typed, multi-field structured output with reusable types:

```yaml
---
model: gpt-5.2
reasoning_effort: high
output:
  result: "number | Final result of the expression"
  explanation: "string | Explanation of the calculation"
  steps: "list[Step] | Ordered calculation steps"
  confidence: "enum[low|medium|high] | Confidence level for the computed answer."

types:
  Step:
    expression: "string | Expression evaluated in this step"
    value: "number | Numeric result of this step"
---
<system>
  You are a helpful assistant and you have access to tools.
  Use tools when needed.
  Return all requested structured fields.
</system>

<user>
  How much is {{expression}}?
</user>
```

Example call:

```python
from prompt_caller import PromptCaller

ai = PromptCaller()

response = ai.call("sample-5.2-complete", {"expression": "(3 + 8) / 9"})
print(response)
```

3. **Using the agent feature:**  

The `agent` method allows you to enhance the prompt's functionality by integrating external tools. Here’s an example where we evaluate a mathematical expression using Python’s `eval` in a safe execution environment:

```python
from prompt_caller import PromptCaller

ai = PromptCaller()

def evaluate_expression(expression: str):
      """
      Evaluate a math expression using eval.
      """
      safe_globals = {"__builtins__": None}
      return eval(expression, safe_globals, {})

response = ai.agent(
      "sample-agent", {"expression": "3+8/9"}, tools=[evaluate_expression]
)

print(response)
```

In this example:

- The `agent` method is used to process the prompt while integrating external tools.
- The `evaluate_expression` function evaluates the mathematical expression securely.
- The response includes the processed result based on the prompt and tool execution.


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


## Tests

```
pytest --cov=prompt_caller ; coverage report --sort=miss
```

## README TODO (Remaining Improvements)

- Document `call()` vs `agent()` precedence explicitly: prompt `output` is used by default, but `agent(..., output=...)` overrides it.
- Add one `agent()` example with real tools and one with explicit Pydantic `output` override.
- Add an image example using `<image>` blocks (URL and data URL forms).
- Add a concise error troubleshooting section for malformed output schemas (unknown type, invalid enum descriptions, bad optional syntax).
- Fix text encoding artifacts in README (for example `Hereâ€™s` should be `Here's`).

## Output Schema DSL

The `output` field supports both legacy and typed schema definitions.

Legacy format (defaults to `string`):

```yaml
output:
  answer: "Final answer to return"
```

Compact DSL format (recommended):

```yaml
output:
  title: "string | Final title"
  confidence: "enum[low|medium|high] | Confidence level"
  steps: "list[Step] | Ordered calculation steps"
  note?: "string | Optional extra note"
```

Supported type expressions:

- `string`
- `number`
- `integer`
- `boolean`
- `list[T]`
- `enum[a|b|c]`
- named type references declared under top-level `types`

Named reusable types:

```yaml
types:
  Step:
    expression: "string | Expression evaluated in this step"
    value: "number | Numeric result of this step"
```

Rules:

- Optional fields are declared with `?` suffix (for example `note?`).
- `call()` uses prompt `output` when present.
- `agent()` uses prompt `output` only when `output=` is not explicitly passed.
