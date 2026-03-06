---
name: prompt-caller
description: Use this skill when creating or editing PromptCaller prompts and integrations. Covers prompt format (YAML frontmatter + JSX-like blocks + Jinja templating), output schema DSL and legacy compatibility, tool-enabled agent usage, and configuration/override rules.
---

# PromptCaller Skill

Use PromptCaller with three building blocks:
- Frontmatter YAML (`model`, model args, `output`, optional `types`)
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
