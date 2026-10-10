import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

import prompt_caller.prompt_caller as pc_module


class ExplicitOutput(BaseModel):
    code: str


def get_course_code() -> str:
    """Look up the current course code."""
    return "QWEN-COURSE-7319"


@pytest.mark.parametrize("explicit_output", [False, True])
@pytest.mark.parametrize("has_system", [False, True])
def test_ollama_formats_complete_tool_history_after_agent(
    prompt_caller, write_prompt, monkeypatch, explicit_output, has_system
):
    write_prompt(
        "structured_agent",
        """
provider: ollama
model: qwen3.5:4b-q4_K_M
output:
  steps: "list[Step] | Tool results"
types:
  Step:
    code: "string | Exact code"
""",
        ('<system>Use the course tool.</system>' if has_system else '')
        + '<user>What is the code for {{course}}?</user>',
    )
    history = [
        HumanMessage(content="What is the code for Qwen?"),
        AIMessage(
            content="",
            tool_calls=[{"name": "get_course_code", "args": {}, "id": "lookup"}],
        ),
        ToolMessage(content="QWEN-COURSE-7319", tool_call_id="lookup"),
        AIMessage(content="The code is QWEN-COURSE-7319."),
    ]
    original_history = list(history)
    captured = {}
    events = []

    class ChatStub:
        def with_structured_output(self, schema):
            events.append("bind_schema")
            captured["schema"] = schema
            return self

        def invoke(self, messages):
            events.append("format")
            captured["format_messages"] = messages
            data = (
                {"code": "QWEN-COURSE-7319"}
                if explicit_output
                else {"steps": [{"code": "QWEN-COURSE-7319"}]}
            )
            return captured["schema"].model_validate(data)

    chat = ChatStub()

    class GraphStub:
        def invoke(self, payload, config):
            events.append("agent")
            captured["payload"] = payload
            captured["config"] = config
            return {"messages": history}

    def create_agent(**kwargs):
        captured["agent_kwargs"] = kwargs
        return GraphStub()

    monkeypatch.setattr(pc_module, "ChatOllama", lambda **_: chat)
    monkeypatch.setattr(pc_module, "create_agent", create_agent)
    response = prompt_caller.agent(
        "structured_agent",
        {"course": "Qwen"},
        tools=[get_course_code],
        output=ExplicitOutput if explicit_output else None,
        allowed_steps=7,
    )

    assert events == ["agent", "bind_schema", "format"]
    assert captured["agent_kwargs"]["response_format"] is None
    assert captured["agent_kwargs"]["model"] is chat
    assert [t.name for t in captured["agent_kwargs"]["tools"]] == ["get_course_code"]
    assert captured["config"] == {"recursion_limit": 7}
    assert captured["payload"]["messages"][0].content == history[0].content
    final_messages = captured["format_messages"]
    if has_system:
        assert isinstance(final_messages[0], SystemMessage)
        assert final_messages[0].content == "Use the course tool."
        final_messages = final_messages[1:]
    assert final_messages[:-1] == history
    assert isinstance(final_messages[-1], HumanMessage)
    assert "Do not add information or call tools" in final_messages[-1].content
    assert history == original_history
    assert isinstance(response, BaseModel)
    if explicit_output:
        assert captured["schema"] is ExplicitOutput
        assert response.code == "QWEN-COURSE-7319"
    else:
        assert response.steps[0].code == "QWEN-COURSE-7319"


def test_ollama_formatting_error_is_not_replaced_with_agent_text(
    prompt_caller, write_prompt, monkeypatch
):
    write_prompt(
        "invalid_output",
        'provider: ollama\nmodel: qwen3.5:4b-q4_K_M\noutput:\n  code: "string"',
        '<user>Look up the course code.</user>',
    )

    class ChatStub:
        def with_structured_output(self, schema):
            return self

        def invoke(self, messages):
            raise OutputParserException("Invalid structured response")

    class GraphStub:
        def invoke(self, payload, config):
            return {"messages": [AIMessage(content="unvalidated answer")]}

    monkeypatch.setattr(pc_module, "ChatOllama", lambda **_: ChatStub())
    monkeypatch.setattr(pc_module, "create_agent", lambda **_: GraphStub())

    with pytest.raises(OutputParserException, match="Invalid structured response"):
        prompt_caller.agent("invalid_output", tools=[get_course_code])


@pytest.mark.parametrize(
    "provider,has_tools,has_schema",
    [
        ("openai", True, True),
        ("google_genai", True, True),
        ("ollama", False, True),
        ("ollama", True, False),
    ],
)
def test_other_agent_paths_do_not_add_a_formatting_call(
    prompt_caller, write_prompt, monkeypatch, provider, has_tools, has_schema
):
    write_prompt(
        "unchanged",
        f"provider: {provider}\nmodel: test-model",
        '<user>Look up the course code.</user>',
    )
    captured = {}
    expected = ExplicitOutput(code="QWEN-COURSE-7319") if has_schema else AIMessage(content="ok")

    class ChatStub:
        def with_structured_output(self, schema):
            pytest.fail("This path must not add a formatting call")

    class GraphStub:
        def invoke(self, payload, config):
            return {"structured_response": expected, "messages": [expected]}

    def create_agent(**kwargs):
        captured.update(kwargs)
        return GraphStub()

    provider_class = {
        "openai": "ChatOpenAI",
        "google_genai": "ChatGoogleGenerativeAI",
        "ollama": "ChatOllama",
    }[provider]
    monkeypatch.setattr(pc_module, provider_class, lambda **_: ChatStub())
    monkeypatch.setattr(pc_module, "create_agent", create_agent)

    assert prompt_caller.agent(
        "unchanged",
        tools=[get_course_code] if has_tools else [],
        output=ExplicitOutput if has_schema else None,
    ) is expected
    assert captured["response_format"] is (ExplicitOutput if has_schema else None)
