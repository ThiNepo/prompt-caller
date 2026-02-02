# This is can be used to showcase a concrete example using your library.

from prompt_caller import PromptCaller

ai = PromptCaller()


def evaluate_expression(expression: str):
    """
    Evaluate a math expression using eval.
    """
    safe_globals = {"__builtins__": None}
    return eval(expression, safe_globals, {})


# response = ai.agent(
#     "sample-5.2-pro", {"expression": "3+8/9"}, tools=[evaluate_expression]
# )

response = ai.call("sample-image")

print(response)
