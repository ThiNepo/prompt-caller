from prompt_caller import PromptCaller

ai = PromptCaller()

response = ai.call("sample", {"expression": "3+8/9"})

print(response)
