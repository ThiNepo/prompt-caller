from prompt_caller import PromptCaller

ai = PromptCaller()

# response = ai.call(
#     "comment_to_thread",
#     {
#         "comment_id": 123,
#         "text": "Esse comentário deve ser traduzido para vários idiomas.",
#     },
# )

# print("GB:", response.message.gb)
# print("BR:", response.message.br)
# print("Message dict:", response.message.model_dump())


def evaluate_expression(expression: str):
    """
    Evaluate a math expression using eval.
    """
    safe_globals = {"__builtins__": None}
    return eval(expression, safe_globals, {})


response = ai.agent("sample-5.2", {"expression": "3+8/9"}, tools=[evaluate_expression])

# response = ai.call("sample-image")

print(response)
