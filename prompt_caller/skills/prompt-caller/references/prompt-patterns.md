# Prompt Patterns

## 1) Simple call with legacy output

```yaml
---
model: gpt-4o-mini
output:
  answer: "Final answer"
---
<user>
Answer: {{question}}
</user>
```

## 2) Typed output with DSL

```yaml
---
model: gpt-5.2
output:
  result: "number | Final computed value"
  confidence:
    description: "enum[low|medium|high] | Confidence level."
    enum_descriptions:
      low: "Uncertain"
      medium: "Likely correct"
      high: "Verified"
types:
  Step:
    expression: "string | Expression evaluated"
    value: "number | Result value"
---
<user>
Compute {{expression}} and provide confidence.
</user>
```

## 3) Agent prompt with tools

```yaml
---
model: gpt-5.2
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

## 4) Images

Use `<image>` block with either:
- URL (PromptCaller fetches + converts to base64)
- Inline data URL

```xml
<image tag="chart">
https://example.com/chart.png
</image>
```
