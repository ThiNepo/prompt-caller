---
name: prompt-caller
description: Use this skill when creating or editing PromptCaller prompts and integrations. Covers prompt format (YAML frontmatter + JSX-like blocks + Jinja templating), output schema DSL and legacy compatibility, tool-enabled agent usage, and configuration/override rules.
---

# PromptCaller Skill

Use PromptCaller with three building blocks:
- Frontmatter YAML (`model`, optional `provider`, model args, `output`, optional `types`)
- Prompt body (`<system>`, `<user>`, `<image>` blocks)
- Jinja interpolation (`{{variable}}`) in body content

## Prompt Structure

Write prompts using this envelope:

```yaml
---
model: gpt-5.2
reasoning_effort: high
output:
  result: "number | Final numeric result"
  explanation: "string | Explanation"
---
<system>
You are a helpful assistant.
</system>

<user>
Solve {{ expression }}.
</user>
```

## Model Providers

- Set `provider` to `openai`, `google_genai`, or `ollama` to choose the chat client.
- If omitted, model names starting with `gemini` use Google; other names use OpenAI.
- `provider`, `output`, and `types` are internal metadata, not client parameters.
- Use options supported by the selected client. Ollama accepts `base_url`,
  `temperature`, `num_ctx`, and `reasoning`; do not copy OpenAI-only options.
- For the course, use `provider: ollama` and `model: qwen3.5:4b-q4_K_M`.
  Start Ollama and run `ollama pull qwen3.5:4b-q4_K_M` first.
- Both `call()` and `agent()` select the provider from the prompt. Tool use
  requires a model that supports tool calling.
- Ollama `agent()` with tools and a schema runs tools first, then formats the full
  conversation in one extra model call without bound tools. Prompt `output` and
  explicit `output=` are supported; formatting errors propagate, not plain text.
  OpenAI/Gemini and other Ollama paths retain their existing behavior.

## Output Schema Rules

`output` supports:
- Legacy: `field: "Description"` (defaults to `str`)
- Compact DSL: `field: "type | description"`
- Typed dict:
  - `type`
  - `description`
  - `enum_descriptions` (only with `enum[...]`)

Supported type expressions:
- `string`, `number`, `integer`, `boolean`
- `list[T]`
- `enum[a|b|c]`
- Named type references declared in top-level `types`

Optional fields:
- Use key suffix `?`, for example `notes?`

## Call vs Agent

- `call(prompt, context)`:
  - model invocation from prompt messages
  - prompt `output` drives structured output if present
- `agent(prompt, context, tools=..., output=...)`:
  - tool-enabled agent flow
  - explicit `output` argument overrides prompt `output`

## Prompt Writing Checklist

- Keep system instruction focused on behavior and constraints.
- Put variable values in context, not hardcoded in template.
- Add clear output descriptions for model guidance.
- Use `types` for repeated object shapes.
- Keep tool function docstrings explicit so agent knows when to call them.

For extended examples and patterns, read `references/prompt-patterns.md`.
