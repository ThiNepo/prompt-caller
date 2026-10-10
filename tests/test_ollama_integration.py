import os
from uuid import uuid4

import pytest
import yaml
from httpx import HTTPError
from langchain_core.messages import AIMessage
from ollama import Client, ResponseError
from pydantic import BaseModel


MODEL = "qwen3.5:4b-q4_K_M"
pytestmark = pytest.mark.skipif(
    os.getenv("RUN_OLLAMA_TESTS") != "1",
    reason="Set RUN_OLLAMA_TESTS=1 to run against the course model in Ollama.",
)


class CourseOutput(BaseModel):
    code: str


@pytest.fixture(scope="module")
def ollama_url():
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        models = Client(host=base_url, timeout=10).list().models
    except (HTTPError, ResponseError, ConnectionError) as error:
        pytest.fail(f"Ollama is unavailable at {base_url}: {error}", pytrace=False)
    if not any(model.model == MODEL for model in models):
        pytest.fail(
            f"Ollama at {base_url} does not have {MODEL}. Run: ollama pull {MODEL}",
            pytrace=False,
        )
    return base_url


def _configuration(base_url, **metadata):
    return yaml.safe_dump(
        {
            "provider": "ollama",
            "model": MODEL,
            "base_url": base_url,
            "temperature": 0,
            "reasoning": False,
            "num_predict": 512,
            "client_kwargs": {"timeout": 120},
            **metadata,
        },
        sort_keys=False,
    )


def test_call_plain(prompt_caller, write_prompt, ollama_url):
    write_prompt(
        "ollama_plain",
        _configuration(ollama_url),
        "<user>Reply with exactly COURSE_OK.</user>",
    )
    response = prompt_caller.call("ollama_plain")
    assert isinstance(response, AIMessage)
    assert "COURSE_OK" in response.content
    assert response.response_metadata["model"] == MODEL


def test_call_structured_with_named_types(prompt_caller, write_prompt, ollama_url):
    write_prompt(
        "ollama_structured",
        _configuration(
            ollama_url,
            output={
                "result": "number | Final numeric result",
                "confidence": "enum[low|medium|high] | Confidence level",
                "steps": "list[Step] | Calculation steps",
            },
            types={
                "Step": {
                    "expression": "string | Expression evaluated",
                    "value": "number | Numeric result",
                }
            },
        ),
        """
<user>
Calculate 7 + 5.5. Set confidence to high.
Include one step with expression '7 + 5.5' and its numeric value.
</user>
""",
    )
    response = prompt_caller.call("ollama_structured")
    assert response.result == 12.5
    assert response.confidence == "high"
    assert len(response.steps) == 1
    assert response.steps[0].expression == "7 + 5.5"
    assert response.steps[0].value == 12.5


@pytest.mark.parametrize("output_mode", ["none", "prompt", "explicit"])
def test_agent_executes_tool(prompt_caller, write_prompt, ollama_url, output_mode):
    calls = []
    course_code = f"QWEN-COURSE-{uuid4().hex[:8].upper()}"

    def get_course_code() -> str:
        """Return the current course access code. Call this to look up the code."""
        calls.append(True)
        return course_code

    metadata = {}
    if output_mode == "prompt":
        metadata["output"] = {
            "code": "string | Exact code returned by get_course_code"
        }
    elif output_mode == "explicit":
        metadata["output"] = {"ignored": "string | Overridden by output="}
    write_prompt(
        "ollama_agent",
        _configuration(ollama_url, **metadata),
        """
<system>
Call get_course_code before answering. Report the exact code it returns.
</system>
<user>
What is the current course access code?
</user>
""",
    )
    response = prompt_caller.agent(
        "ollama_agent",
        tools=[get_course_code],
        output=CourseOutput if output_mode == "explicit" else None,
    )
    assert calls, "The agent must execute the actual Python tool."
    if output_mode != "none":
        assert isinstance(response, BaseModel) and not isinstance(response, AIMessage), (
            "The agent executed the tool but did not return the requested schema. "
            f"Received {type(response).__name__}: {response}"
        )
        assert response.code == course_code
        if output_mode == "explicit":
            assert isinstance(response, CourseOutput)
    else:
        assert isinstance(response, AIMessage)
        assert course_code in response.content
