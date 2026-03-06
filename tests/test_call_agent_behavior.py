from pydantic import BaseModel
from langchain_core.messages import AIMessage

import prompt_caller.prompt_caller as pc_module


class ExplicitOutput(BaseModel):
    final: str


def _basic_body():
    return """
<system>
You are helpful.
</system>
<user>
Solve {{expression}}.
</user>
"""


def test_call_without_output_uses_langchain_fake_chat_model(
    prompt_caller, write_prompt, monkeypatch, fake_langchain_chat
):
    write_prompt(
        "plain",
        """
model: gpt-5.2
""",
        _basic_body(),
    )

    monkeypatch.setattr(pc_module, "ChatOpenAI", lambda **_: fake_langchain_chat)

    response = prompt_caller.call("plain", {"expression": "1+1"})

    assert isinstance(response, AIMessage)
    assert response.content == "fake-ok"


def test_call_builds_structured_output_and_strips_prompt_metadata(
    prompt_caller, write_prompt, monkeypatch, chat_stub_factory
):
    write_prompt(
        "typed",
        """
model: gpt-5.2
temperature: 0.3
output:
  result: "number | final result"
types:
  Step:
    value: "number"
""",
        _basic_body(),
    )

    chat_kwargs = {}

    def fake_chat_openai(**kwargs):
        chat_kwargs.update(kwargs)
        return chat_stub_factory(response={"result": 2.0})

    model_calls = []

    def fake_create_model(dynamic_dict, type_defs=None):
        model_calls.append((dynamic_dict, type_defs))
        return "DYNAMIC-MODEL"

    monkeypatch.setattr(pc_module, "ChatOpenAI", fake_chat_openai)
    monkeypatch.setattr(prompt_caller, "createPydanticModel", fake_create_model)

    response = prompt_caller.call("typed", {"expression": "1+1"})

    assert response == {"result": 2.0}
    assert chat_kwargs == {"model": "gpt-5.2", "temperature": 0.3}
    assert model_calls[0][0] == {"result": "number | final result"}
    assert model_calls[0][1] == {"Step": {"value": "number"}}
    assert chat_stub_factory.created[0].structured_schema == "DYNAMIC-MODEL"


def test_agent_uses_prompt_output_when_no_explicit_override(
    prompt_caller, write_prompt, monkeypatch, chat_stub_factory
):
    write_prompt(
        "agent_prompt",
        """
model: gpt-5.2
output:
  result: "number | final result"
""",
        _basic_body(),
    )

    monkeypatch.setattr(pc_module, "ChatOpenAI", lambda **_: chat_stub_factory())

    captured = {}

    class AgentGraphStub:
        def invoke(self, payload, config):
            captured["payload"] = payload
            captured["config"] = config
            return {
                "structured_response": {"result": 2.0},
                "messages": [AIMessage(content="unused")],
            }

    def fake_create_agent(**kwargs):
        captured["create_agent_kwargs"] = kwargs
        return AgentGraphStub()

    monkeypatch.setattr(pc_module, "create_agent", fake_create_agent)

    response = prompt_caller.agent("agent_prompt", {"expression": "1+1"})

    assert response == {"result": 2.0}
    assert captured["create_agent_kwargs"]["response_format"] is not None
    assert captured["config"] == {"recursion_limit": 10}


def test_agent_explicit_output_overrides_prompt_output(
    prompt_caller, write_prompt, monkeypatch, chat_stub_factory
):
    write_prompt(
        "agent_override",
        """
model: gpt-5.2
output:
  result: "number | final result"
""",
        _basic_body(),
    )

    monkeypatch.setattr(pc_module, "ChatOpenAI", lambda **_: chat_stub_factory())

    captured = {}

    class AgentGraphStub:
        def invoke(self, payload, config):
            return {
                "structured_response": ExplicitOutput(final="ok"),
                "messages": [AIMessage(content="unused")],
            }

    def fake_create_agent(**kwargs):
        captured["response_format"] = kwargs["response_format"]
        return AgentGraphStub()

    monkeypatch.setattr(pc_module, "create_agent", fake_create_agent)

    response = prompt_caller.agent(
        "agent_override", {"expression": "1+1"}, output=ExplicitOutput
    )

    assert isinstance(response, ExplicitOutput)
    assert captured["response_format"] is ExplicitOutput
