# This is can be used to showcase a concrete example using your library.

from prompt_caller import PromptCaller

ai = PromptCaller()

# response = ai.call("sample-image", {"expression": "3+8/9"})


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
