![PyPI - Python Version](https://img.shields.io/pypi/pyversions/prompt-caller) ![PyPI](https://img.shields.io/pypi/v/prompt-caller)

[![PyPI - Downloads](https://img.shields.io/pypi/dm/prompt-caller)](https://pypi.org/project/prompt-caller/) [![Discord](https://img.shields.io/discord/479923444017004556?label=discord)](https://discord.gg/jWWDRyD5Nu)  [![YouTube Channel Subscribers](https://img.shields.io/youtube/channel/subscribers/UCMb90JgsFJpZyZzdmWCaCTg?style=social)](https://www.youtube.com/channel/UCMb90JgsFJpZyZzdmWCaCTg)

![Logo](images//logo.png)

# PromptCaller

**PromptCaller** centralizes `.prompt` files in one folder, turns prompt responses into structured outputs, and includes SKILLS so agents can create and maintain prompt files consistently.

## Features

- **Load prompts** from a `.prompt` file containing a YAML configuration and a message template.
- **Invoke prompts** using LangChain with OpenAI, Google Gemini, or Ollama, with support for structured output and tools.

## Installation

To install the package, simply run:

```bash
pip install prompt-caller
```

For OpenAI prompts, you will also need an `.env` file that contains your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

### Model Providers

Choose a provider in the `.prompt` frontmatter:

| `provider` | Chat client |
| --- | --- |
| `openai` | `ChatOpenAI` |
| `google_genai` | `ChatGoogleGenerativeAI` |
| `ollama` | `ChatOllama` |

When `provider` is omitted, model names beginning with `gemini` select Google;
all other names select OpenAI. An explicit provider takes precedence over the
model name. `provider`, `output`, and `types` are internal PromptCaller metadata;
the remaining frontmatter configures the selected chat client. Use parameters
supported by that client (for example, `reasoning` for Ollama).

### Local Ollama

Install and start [Ollama](https://ollama.com), then download the course model:

```bash
ollama pull qwen3.5:4b-q4_K_M
```

Use `prompts/sample-ollama.prompt`:

```yaml
---
provider: ollama
model: qwen3.5:4b-q4_K_M
base_url: http://localhost:11434
temperature: 0
reasoning: false
output:
  answer: "string | Answer to the question"
  confidence: "enum[low|medium|high] | Confidence level"
---
<system>
Answer the question clearly and concisely.
Return all requested structured fields.
</system>

<user>
{{question}}
</user>
```

```python
from prompt_caller import PromptCaller

ai = PromptCaller()
response = ai.call("sample-ollama", {"question": "What is 7 + 5?"})
print(response.answer)
```

Local Ollama does not require an OpenAI key. `base_url` is optional; when omitted,
the Ollama client uses its default server address. Tool-enabled agents require a
model that supports tool calling. PromptCaller uses the standard LangChain
structured-output and tool mechanisms.

For Ollama agents with both tools and an output schema, PromptCaller runs the
tool-enabled agent without a response schema, then makes one additional call to
the same model with `with_structured_output()`. The formatting call receives the
system prompt and full conversation, including tool results, without bound tools.
Parsing and validation errors propagate rather than falling back to plain text.
This avoids constraining tool calls with Ollama's JSON schema format. OpenAI and
Gemini keep their existing agent response strategy; other Ollama paths are unchanged.
`allowed_steps` limits the agent graph, not the additional formatting call.

Real tests with `qwen3.5:4b-q4_K_M` passed for structured `call()` output and
`agent()` with actual tool execution, both with and without an output schema.

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

The `agent` method allows you to enhance the prompt's functionality by integrating external tools. Here's an example where we evaluate a mathematical expression using Python�s `eval` in a safe execution environment:

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


## Response Metadata

Both `call()` and `agent()` accept the keyword-only option `include_metadata=False`.
By default their return values and error behavior are unchanged. Set it to `True`
to attach a reserved `_metadata` attribute to the returned object, without a wrapper:

```python
response = ai.call(
    "sample-5.2-complete",
    {"expression": "(3 + 8) / 9"},
    include_metadata=True,
)
print(response.result)  # Same access as before
print(response._metadata["response_id"])
print(response._metadata["response_metadata"])
print(response._metadata["usage_metadata"])

response = ai.agent(
    "sample-agent",
    {"expression": "3+8/9"},
    tools=[evaluate_expression],
    include_metadata=True,
)
print(response._metadata)
```

The attribute always contains these three keys:

| Key | Value |
| --- | --- |
| `response_id` | The provider's `id` or `response_id` from response metadata, or `None`. LangChain-generated message IDs are not used as a fallback. |
| `response_metadata` | An independent deep copy of the provider metadata exposed by LangChain, or `{}` if unavailable. |
| `usage_metadata` | An independent deep copy of LangChain's normalized token usage, or `None` if unavailable. |

The existing object type, prompt schema, `model_dump()`, and `model_dump_json()`
are preserved. When enabled, metadata is attached to a shallow model copy so cached
responses remain untouched; object identity is not preserved in this mode.
`_metadata` is not included in those serialized results. Include it
explicitly when constructing a payload if you need to persist the metadata.
The option belongs in the Python call, not the prompt frontmatter, and does not
add model requests or switch provider APIs. IDs are not necessarily `resp_...`;
for example, OpenAI Chat Completions can return `chatcmpl-...` IDs.

For ordinary agents, the source is the last `AIMessage`, even when followed by a
tool message. If the last AI message has no provider ID, `response_id` is `None`;
an earlier response's ID is not substituted. For Ollama agents with tools and a
schema, the source is the final structured-formatting call. These values do not
represent accumulated usage across agent calls, tools, or retries.

Metadata attachment supports Pydantic `BaseModel` outputs, including `AIMessage`.
When using `agent(output=...)` with this option, pass a Pydantic model class.
Other schema formats (such as `TypedDict`, dataclasses, JSON Schema dictionaries,
and explicit LangChain strategy objects) are rejected before the agent runs;
their existing behavior is unchanged when the option is disabled.
`_metadata` is reserved: conflicting prompt fields, private attributes, properties,
or existing instance attributes raise `ValueError` rather than being overwritten.
Known schema conflicts are checked before model invocation. Unexpected return
types raise `TypeError`. Parsing and provider errors continue to propagate.

## How It Works

1. **\_loadPrompt:** Loads the prompt file, splits the YAML header from the body, and parses them.
2. **\_renderTemplate:** Uses the Jinja2 template engine to render the body with the provided context.
3. **\_parseJSXBody:** Parses the message body written in JSX-like tags to extract system and user messages.
4. **call:** Invokes the selected chat provider with the parsed configuration and messages, and handles structured output via dynamic Pydantic models.

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


## Tests

```
pytest --cov=prompt_caller ; coverage report --sort=miss
```

The regular suite does not contact Ollama. To run the real course-model tests:

```powershell
$env:RUN_OLLAMA_TESTS = "1"
python -m pytest tests/test_ollama_integration.py -v
```

On Bash, use `RUN_OLLAMA_TESTS=1 python -m pytest tests/test_ollama_integration.py -v`.
These tests use exactly `qwen3.5:4b-q4_K_M` and cover plain calls, typed output with
named types, and actual tool execution with and without structured agent output,
including an explicit `output=` override. Tool codes are generated randomly so
the agent must obtain them from the tool rather than infer them from the prompt.
Each real-model scenario runs with response metadata disabled and enabled.
Set `OLLAMA_BASE_URL` to use another server. When enabled, the tests fail with a
clear error if the server is unavailable or the model has not been downloaded.

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

## License

This project is licensed under the **Apache License 2.0**. You may use, modify, and distribute this software as long as you provide proper attribution and include the full text of the license in any distributed copies or derivative works.
