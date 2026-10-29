import json
from copy import deepcopy
from dataclasses import dataclass
from inspect import Parameter, signature

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, ConfigDict, PrivateAttr
from typing_extensions import TypedDict

import prompt_caller.prompt_caller as pc_module


class ExplicitOutput(BaseModel):
    answer: str


class FrozenOutput(ExplicitOutput):
    model_config = ConfigDict(frozen=True)


class DictionaryOutput(TypedDict):
    answer: str


@dataclass
class DataclassOutput:
    answer: str


class ConflictingOutput(ExplicitOutput):
    _metadata: dict = PrivateAttr(default_factory=lambda: {"owned_by": "user"})


class MetadataPropertyOutput(ExplicitOutput):
    @property
    def _metadata(self):
        return {"owned_by": "user"}


def _raw_response(response_id="resp_final"):
    metadata = {"model_name": "test-model", "details": {"finish_reason": "stop"}}
    if response_id is not None:
        metadata["id"] = response_id
    return AIMessage(
        content='{"answer":"ok"}',
        id="lc_run--not-a-provider-id",
        response_metadata=metadata,
        usage_metadata={
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_token_details": {"cache_read": 2},
        },
    )


class MetadataChatStub:
    def __init__(self, raw):
        self.raw = raw
        self.schema = None
        self.schema_kwargs = None
        self.parsing_error = None
        self.parsed = None
        self.invocations = []

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        self.schema_kwargs = kwargs
        return self

    def invoke(self, messages):
        self.invocations.append(messages)
        if self.schema is None:
            return self.raw
        self.parsed = (
            None if self.parsing_error else self.schema.model_validate({"answer": "ok"})
        )
        if self.schema_kwargs.get("include_raw"):
            return {
                "parsed": self.parsed,
                "raw": self.raw,
                "parsing_error": self.parsing_error,
            }
        if self.parsing_error:
            raise self.parsing_error
        return self.parsed


def _write_prompt(write_prompt, provider="openai", structured=False):
    frontmatter = f"provider: {provider}\nmodel: test-model"
    if structured:
        frontmatter += '\noutput:\n  answer: "string | Final answer"'
    write_prompt("metadata", frontmatter, "<user>Answer {{question}}.</user>")


def _install_graph(monkeypatch, raw, parsed=None, messages=None, error=None):
    captured = {}

    class GraphStub:
        def invoke(self, payload, config):
            captured["invoke"] = (payload, config)
            if error:
                raise error
            return {
                "messages": messages if messages is not None else [raw],
                "structured_response": parsed,
            }

    def create_agent(**kwargs):
        captured.update(kwargs)
        return GraphStub()

    monkeypatch.setattr(pc_module, "create_agent", create_agent)
    return captured


@pytest.mark.parametrize("method", ["call", "agent"])
@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("metadata_kwargs", [{}, {"include_metadata": False}, {"include_metadata": True}])
def test_metadata_is_opt_in_and_preserves_response(
    prompt_caller, write_prompt, monkeypatch, method, structured, metadata_kwargs
):
    _write_prompt(write_prompt, structured=structured)
    raw = _raw_response()
    raw_dump = deepcopy(raw.model_dump())
    chat = MetadataChatStub(raw)
    monkeypatch.setattr(prompt_caller, "_createChat", lambda _: chat)
    expected = ExplicitOutput(answer="ok") if structured else raw
    _install_graph(monkeypatch, raw, expected if structured else None)

    response = getattr(prompt_caller, method)(
        "metadata", {"question": "this"}, **metadata_kwargs
    )

    if method == "call" and structured:
        expected = chat.parsed
        assert chat.schema_kwargs == (
            {"include_raw": True} if metadata_kwargs.get("include_metadata") else {}
        )
    if metadata_kwargs.get("include_metadata"):
        assert type(response) is type(expected)
        assert response is not expected
        assert not hasattr(expected, "_metadata")
    else:
        assert response is expected
    assert response.model_dump() == (raw_dump if not structured else {"answer": "ok"})
    assert "_metadata" not in type(response).model_json_schema().get("properties", {})
    assert "_metadata" not in json.loads(response.model_dump_json())
    if metadata_kwargs.get("include_metadata"):
        assert response._metadata == {
            "response_id": "resp_final",
            "response_metadata": raw.response_metadata,
            "usage_metadata": raw.usage_metadata,
        }
        assert response._metadata["response_metadata"] is not raw.response_metadata
        assert response._metadata["usage_metadata"] is not raw.usage_metadata
        response._metadata["response_metadata"]["details"]["finish_reason"] = "changed"
        response._metadata["usage_metadata"]["input_token_details"]["cache_read"] = 99
        assert raw.response_metadata["details"]["finish_reason"] == "stop"
        assert raw.usage_metadata["input_token_details"]["cache_read"] == 2
    else:
        assert not hasattr(response, "_metadata")


@pytest.mark.parametrize("provider", ["openai", "google_genai", "ollama"])
@pytest.mark.parametrize("structured", [False, True])
def test_call_metadata_is_provider_independent(
    prompt_caller, write_prompt, monkeypatch, provider, structured
):
    _write_prompt(write_prompt, provider=provider, structured=structured)
    raw = _raw_response("chatcmpl-example" if provider == "openai" else None)
    raw.usage_metadata = None
    chat = MetadataChatStub(raw)
    captured = {}

    def create_chat(**kwargs):
        captured.update(kwargs)
        return chat

    provider_class = {
        "openai": "ChatOpenAI", "google_genai": "ChatGoogleGenerativeAI", "ollama": "ChatOllama"
    }[provider]
    monkeypatch.setattr(pc_module, provider_class, create_chat)
    response = prompt_caller.call("metadata", include_metadata=True)
    assert response._metadata["response_id"] == raw.response_metadata.get("id")
    assert response._metadata["usage_metadata"] is None
    assert "include_metadata" not in captured


@pytest.mark.parametrize("method", ["call", "agent"])
def test_provider_response_id_key_is_preserved(prompt_caller, write_prompt, monkeypatch, method):
    _write_prompt(write_prompt)
    raw = _raw_response(None)
    raw.response_metadata["response_id"] = "provider-specific-id"
    monkeypatch.setattr(prompt_caller, "_createChat", lambda _: MetadataChatStub(raw))
    _install_graph(monkeypatch, raw)
    response = getattr(prompt_caller, method)("metadata", include_metadata=True)
    assert response._metadata["response_id"] == "provider-specific-id"


@pytest.mark.parametrize("has_ai_message", [False, True])
def test_agent_uses_last_ai_message_not_last_tool_or_last_id(
    prompt_caller, write_prompt, monkeypatch, has_ai_message
):
    _write_prompt(write_prompt, structured=True)
    older = _raw_response("resp_older")
    latest = _raw_response(None)
    trailing_tool = ToolMessage(content="ack", tool_call_id="structured")
    parsed = ExplicitOutput(answer="ok")
    messages = [older, latest, trailing_tool] if has_ai_message else [trailing_tool]
    monkeypatch.setattr(prompt_caller, "_createChat", lambda _: MetadataChatStub(latest))
    _install_graph(monkeypatch, latest, parsed, messages=messages)
    response = prompt_caller.agent("metadata", include_metadata=True)
    assert type(response) is ExplicitOutput
    assert response is not parsed
    assert response._metadata["response_id"] is None
    assert response._metadata["response_metadata"] == (
        latest.response_metadata if has_ai_message else {}
    )
    assert response._metadata["usage_metadata"] == (
        latest.usage_metadata if has_ai_message else None
    )


@pytest.mark.parametrize("provider", ["openai", "google_genai", "ollama"])
@pytest.mark.parametrize("schema", [ExplicitOutput, FrozenOutput])
def test_agent_metadata_respects_explicit_output_and_ollama_final_call(
    prompt_caller, write_prompt, monkeypatch, provider, schema
):
    _write_prompt(write_prompt, provider=provider, structured=True)
    final_raw = _raw_response("resp_format")
    graph_raw = _raw_response("resp_agent")
    graph_raw.usage_metadata = {
        "input_tokens": 100, "output_tokens": 90, "total_tokens": 190
    }
    chat = MetadataChatStub(final_raw)
    monkeypatch.setattr(prompt_caller, "_createChat", lambda _: chat)
    parsed = schema(answer="ok")
    captured = _install_graph(monkeypatch, graph_raw, parsed)

    def lookup() -> str:
        """Look up the answer."""
        return "ok"

    response = prompt_caller.agent(
        "metadata", tools=[lookup], output=schema, include_metadata=True
    )
    assert type(response) is schema
    assert response.answer == "ok"
    if provider == "ollama":
        assert captured["response_format"] is None
        assert chat.schema is schema
        assert chat.schema_kwargs == {"include_raw": True}
        assert len(chat.invocations) == 1
        assert graph_raw in chat.invocations[0]
        assert response._metadata["response_id"] == "resp_format"
        assert response._metadata["usage_metadata"]["total_tokens"] == 15
    else:
        assert captured["response_format"] is schema
        assert not chat.invocations
        assert response is not parsed
        assert response._metadata["response_id"] == "resp_agent"
        assert response._metadata["usage_metadata"]["total_tokens"] == 190
    assert response.model_dump() == {"answer": "ok"}


@pytest.mark.parametrize("method", ["call", "ollama_agent"])
@pytest.mark.parametrize("include_metadata", [False, True])
def test_structured_parsing_errors_keep_raising(
    prompt_caller, write_prompt, monkeypatch, method, include_metadata
):
    _write_prompt(write_prompt, provider="ollama", structured=True)
    chat = MetadataChatStub(_raw_response())
    error = OutputParserException("Invalid structured response")
    chat.parsing_error = error
    monkeypatch.setattr(prompt_caller, "_createChat", lambda _: chat)
    _install_graph(monkeypatch, _raw_response())

    def lookup() -> str:
        """Look up the answer."""
        return "ok"

    with pytest.raises(OutputParserException) as raised:
        if method == "call":
            prompt_caller.call("metadata", include_metadata=include_metadata)
        else:
            prompt_caller.agent("metadata", tools=[lookup], include_metadata=include_metadata)
    assert raised.value is error


@pytest.mark.parametrize("schema", [DictionaryOutput, DataclassOutput, {"type": "object"}])
def test_unsupported_metadata_schema_fails_before_chat_creation(
    prompt_caller, write_prompt, monkeypatch, schema
):
    _write_prompt(write_prompt)
    monkeypatch.setattr(
        prompt_caller, "_createChat", lambda _: pytest.fail("Must reject before creating chat")
    )
    with pytest.raises(TypeError, match="requires a Pydantic BaseModel"):
        prompt_caller.agent("metadata", output=schema, include_metadata=True)


@pytest.mark.parametrize("schema", [DictionaryOutput, DataclassOutput, {"type": "object"}])
def test_unsupported_metadata_schema_still_works_without_option(
    prompt_caller, write_prompt, monkeypatch, schema
):
    _write_prompt(write_prompt)
    raw = _raw_response()
    parsed = {"answer": "ok"}
    monkeypatch.setattr(prompt_caller, "_createChat", lambda _: MetadataChatStub(raw))
    captured = _install_graph(monkeypatch, raw, parsed)
    assert prompt_caller.agent("metadata", output=schema) is parsed
    assert captured["response_format"] is schema


@pytest.mark.parametrize("schema", [ConflictingOutput, MetadataPropertyOutput])
def test_metadata_name_collision_in_explicit_schema_is_rejected_early(
    prompt_caller, write_prompt, monkeypatch, schema
):
    _write_prompt(write_prompt)
    monkeypatch.setattr(
        prompt_caller, "_createChat", lambda _: pytest.fail("Must reject before creating chat")
    )
    with pytest.raises(ValueError, match="reserved for PromptCaller"):
        prompt_caller.agent("metadata", output=schema, include_metadata=True)


@pytest.mark.parametrize("method", ["call", "agent"])
@pytest.mark.parametrize("field", ["_metadata", "_metadata?"])
def test_metadata_name_in_prompt_schema_is_reserved(
    prompt_caller, write_prompt, monkeypatch, method, field
):
    write_prompt("metadata", f'model: test\noutput:\n  {field}: "string"', '<user>Hi</user>')
    monkeypatch.setattr(
        prompt_caller, "_createChat", lambda _: pytest.fail("Must reject before creating chat")
    )
    with pytest.raises(ValueError, match="reserved for PromptCaller"):
        getattr(prompt_caller, method)("metadata", include_metadata=True)


@pytest.mark.parametrize("method", ["call", "agent"])
@pytest.mark.parametrize("existing_metadata", [None, {"owned_by": "user"}])
def test_existing_instance_metadata_is_not_overwritten(
    prompt_caller, write_prompt, monkeypatch, method, existing_metadata
):
    _write_prompt(write_prompt)
    raw = _raw_response()
    raw._metadata = existing_metadata
    monkeypatch.setattr(prompt_caller, "_createChat", lambda _: MetadataChatStub(raw))
    _install_graph(monkeypatch, raw)
    with pytest.raises(ValueError, match="reserved for PromptCaller"):
        getattr(prompt_caller, method)("metadata", include_metadata=True)
    assert raw._metadata is existing_metadata


def test_explicit_output_overrides_reserved_name_in_unused_prompt_schema(
    prompt_caller, write_prompt, monkeypatch
):
    write_prompt(
        "metadata", 'model: test\noutput:\n  _metadata: "string"', '<user>Hi</user>'
    )
    raw = _raw_response()
    parsed = ExplicitOutput(answer="ok")
    monkeypatch.setattr(prompt_caller, "_createChat", lambda _: MetadataChatStub(raw))
    _install_graph(monkeypatch, raw, parsed)
    response = prompt_caller.agent("metadata", output=ExplicitOutput, include_metadata=True)
    assert type(response) is ExplicitOutput
    assert response is not parsed
    assert response._metadata["response_id"] == "resp_final"


@pytest.mark.parametrize("include_metadata", [False, True])
def test_agent_errors_propagate_unchanged(
    prompt_caller, write_prompt, monkeypatch, include_metadata
):
    _write_prompt(write_prompt, structured=True)
    raw = _raw_response()
    monkeypatch.setattr(prompt_caller, "_createChat", lambda _: MetadataChatStub(raw))
    error = RuntimeError("Provider failed")
    _install_graph(monkeypatch, raw, error=error)
    with pytest.raises(RuntimeError) as raised:
        prompt_caller.agent("metadata", include_metadata=include_metadata)
    assert raised.value is error


def test_metadata_is_keyword_only(prompt_caller):
    for method in [prompt_caller.call, prompt_caller.agent]:
        parameter = signature(method).parameters["include_metadata"]
        assert parameter.kind is Parameter.KEYWORD_ONLY
        assert parameter.default is False
