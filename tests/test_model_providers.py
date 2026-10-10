from copy import deepcopy

import pytest
from langchain_core.messages import AIMessage

import prompt_caller.prompt_caller as pc_module


@pytest.mark.parametrize(
    "configuration,expected_provider",
    [
        ({"model": "gpt-5.2"}, "ChatOpenAI"),
        ({"model": "gemini-2.5-flash"}, "ChatGoogleGenerativeAI"),
        ({"model": "qwen3.5:4b-q4_K_M"}, "ChatOpenAI"),
        ({"provider": "openai", "model": "gemini-custom"}, "ChatOpenAI"),
        ({"provider": "google_genai", "model": "custom"}, "ChatGoogleGenerativeAI"),
        ({"provider": "ollama", "model": "gemini-custom"}, "ChatOllama"),
    ],
)
def test_provider_selection_preserves_configuration(
    prompt_caller, monkeypatch, configuration, expected_provider
):
    configuration.update(
        temperature=0,
        output={"steps": "list[Step]"},
        types={"Step": {"value": "number"}},
    )
    original = deepcopy(configuration)
    captured = {}

    def make_factory(provider):
        def factory(**kwargs):
            captured.update(provider=provider, kwargs=kwargs)
            return provider

        return factory

    for name in ("ChatOpenAI", "ChatGoogleGenerativeAI", "ChatOllama"):
        monkeypatch.setattr(pc_module, name, make_factory(name))

    assert prompt_caller._createChat(configuration) == expected_provider
    assert captured["kwargs"] == {
        "model": configuration["model"],
        "temperature": 0,
    }
    assert configuration == original


@pytest.mark.parametrize("provider", ["unknown", "", None, 123, ["ollama"]])
def test_invalid_provider_has_clear_error(prompt_caller, provider):
    with pytest.raises(ValueError, match="Expected one of: openai, google_genai, ollama"):
        prompt_caller._createChat({"provider": provider, "model": "qwen3.5:4b-q4_K_M"})


@pytest.mark.parametrize("method", ["call", "agent"])
def test_ollama_receives_model_options_and_keeps_named_output_types(
    prompt_caller, write_prompt, monkeypatch, chat_stub_factory, method
):
    write_prompt(
        "ollama",
        """
provider: ollama
model: qwen3.5:4b-q4_K_M
base_url: http://localhost:11434
temperature: 0
num_ctx: 4096
output:
  steps: "list[Step] | Calculation steps"
types:
  Step:
    value: "number | Result"
""",
        "<user>Solve {{expression}}.</user>",
    )
    captured = {}
    response = {"steps": [{"value": 2.0}]}

    def fake_ollama(**kwargs):
        captured["model_kwargs"] = kwargs
        return chat_stub_factory(response=response)

    class AgentGraphStub:
        def invoke(self, payload, config):
            return {
                "structured_response": response,
                "messages": [AIMessage(content="unused")],
            }

    def fake_create_agent(**kwargs):
        captured["schema"] = kwargs["response_format"]
        assert kwargs["model"] is chat_stub_factory.created[0]
        return AgentGraphStub()

    monkeypatch.setattr(pc_module, "ChatOllama", fake_ollama)
    monkeypatch.setattr(pc_module, "create_agent", fake_create_agent)

    assert getattr(prompt_caller, method)("ollama", {"expression": "1+1"}) == response
    assert captured["model_kwargs"] == {
        "model": "qwen3.5:4b-q4_K_M",
        "base_url": "http://localhost:11434",
        "temperature": 0,
        "num_ctx": 4096,
    }
    if method == "agent":
        schema = captured["schema"]
    else:
        schema = chat_stub_factory.created[0].structured_schema
    assert schema.model_validate(response).steps[0].value == 2.0
