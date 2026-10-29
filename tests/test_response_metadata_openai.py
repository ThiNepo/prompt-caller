import httpx
import pytest
from copy import deepcopy
from langchain_core.messages import AIMessage
from langchain_core.caches import InMemoryCache
from langchain_openai import ChatOpenAI

import prompt_caller.prompt_caller as pc_module


@pytest.mark.parametrize("method", ["call", "agent"])
@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("responses_api", [False, True])
@pytest.mark.parametrize("cached", [False, True])
def test_metadata_with_real_openai_sdk_and_offline_transport(
    prompt_caller, write_prompt, monkeypatch, method, structured, responses_api, cached
):
    frontmatter = (
        "model: gpt-4o-mini\n"
        f"use_responses_api: {str(responses_api).lower()}"
    )
    if structured:
        frontmatter += '\noutput:\n  answer: "string | Final answer"'
    write_prompt("offline", frontmatter, "<user>Say ok.</user>")
    response_id = "resp_offline" if responses_api else "chatcmpl-offline"
    text = '{"answer":"ok"}' if structured else "ok"
    requests = []

    def handle(request):
        requests.append(request)
        if responses_api:
            body = {
                "id": response_id,
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-4o-mini",
                "output": [{
                    "type": "message",
                    "id": "msg_offline",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            }
        else:
            body = {
                "id": response_id,
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": text},
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        return httpx.Response(200, json=body)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        cache = InMemoryCache() if cached else False

        def create_chat(**kwargs):
            return ChatOpenAI(api_key="offline-test", http_client=client, cache=cache, **kwargs)

        monkeypatch.setattr(pc_module, "ChatOpenAI", create_chat)
        response = getattr(prompt_caller, method)("offline", include_metadata=True)
        if cached:
            original_metadata = deepcopy(response._metadata)
            repeated = getattr(prompt_caller, method)("offline", include_metadata=True)
            assert repeated._metadata["response_id"] == response_id
            assert repeated._metadata["response_metadata"] == response._metadata["response_metadata"]
            assert repeated._metadata["usage_metadata"]["total_tokens"] == 15
            assert repeated._metadata is not response._metadata
            plain = getattr(prompt_caller, method)("offline")
            assert not hasattr(plain, "_metadata")
            assert response._metadata == original_metadata

    if structured:
        assert response.answer == "ok"
    else:
        assert isinstance(response, AIMessage)
    assert response._metadata["response_id"] == response_id
    assert response._metadata["response_id"] != "msg_offline"
    assert response._metadata["usage_metadata"]["total_tokens"] == 15
    assert "_metadata" not in response.model_dump()
    assert len(requests) == 1
    expected_path = "/responses" if responses_api else "/chat/completions"
    assert requests[0].url.path.endswith(expected_path)
