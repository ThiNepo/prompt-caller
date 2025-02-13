# This is can be used to showcase a concrete example using your library.

from prompt_caller import PromptCaller

ai = PromptCaller()

response = ai.call("sample-o3", {"expression": "3+8/9"})

print(response)
