# Prompt Patterns

Use these patterns as a starting point. Prefer the compact DSL form:

```yaml
output:
  answer: "string | Final answer"
```

Use top-level `types` only when a field actually references a named type like `list[Step]` or `Step`.

## 1) Simple call with legacy output

```yaml
---
model: gpt-5.4-mini
output:
  answer: "Final answer"
---
<user>
Answer: {{question}}
</user>
```

## 2) Typed output with compact DSL

```yaml
---
model: gpt-5.4
reasoning_effort: high
output:
  result: "number | Final computed value"
  confidence: "enum[low|medium|high] | Confidence level."
---
<user>
Compute {{expression}} and provide confidence.
</user>
```

## 3) Typed output with reusable named type

```yaml
---
model: gpt-5.4
reasoning_effort: high
output:
  result: "number | Final result"
  steps: "list[Step] | Ordered calculation steps"
types:
  Step:
    expression: "string | Expression evaluated in this step"
    value: "number | Numeric result of this step"
---
<user>
Solve {{expression}} and show the intermediate steps.
</user>
```

## 4) Agent prompt with tools

```yaml
---
model: gpt-5.4
output:
  result: "number | Final result"
  explanation: "string | Tool-backed explanation"
---
<system>
Use available tools before finalizing.
</system>

<user>
How much is {{expression}}?
</user>
```

## 5) Images

Use `<image>` block with either:
- URL (PromptCaller fetches + converts to base64)
- Inline data URL

```xml
<image tag="chart">
https://example.com/chart.png
</image>
```

Example with prompt body:

```xml
<system>
Review the attached chart and summarize the trend.
</system>

<image tag="sales-chart">
https://example.com/chart.png
</image>

<user>
What changed from left to right?
</user>
```

## 6) Local Ollama with the course model

Start Ollama and run `ollama pull qwen3.5:4b-q4_K_M` before invoking this prompt.
Save it as `sample-ollama.prompt`:

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

The prompt determines the provider for both `call()` and `agent()`.
For tool-enabled course agents, keep `output` and call
`ai.agent(prompt_name, context, tools=[...])`. PromptCaller executes the tools
without a response schema, then formats the conversation and tool results with
one additional structured model call. An explicit `output=` overrides the prompt
schema. Formatting errors propagate instead of returning unvalidated text.
The course model passed real tests for both structured calls and tool-enabled
structured agents. `base_url` can be omitted
to use the Ollama client default. Local Ollama does not require an OpenAI key.
`reasoning: false` disables thinking for this short example; it is an Ollama
option, whereas `reasoning_effort` belongs to the OpenAI client.
