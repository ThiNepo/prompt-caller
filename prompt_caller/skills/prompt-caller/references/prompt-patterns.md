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
model: gpt-4o-mini
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
model: gpt-5.2
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
model: gpt-5.2
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
