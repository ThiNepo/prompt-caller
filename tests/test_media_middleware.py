from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command


class RequestStub:
    def __init__(self, tool_call_id="tool-123"):
        self.tool_call = {"id": tool_call_id}


class ResultStub:
    def __init__(self, content):
        self.content = content


def test_media_middleware_wraps_image_payload(prompt_caller):
    middleware = prompt_caller._create_media_middleware()

    request = RequestStub("img-1")

    def handler(_request):
        return ResultStub(content=[{"image_url": {"url": "data:image/png;base64,AAA"}}])

    result = middleware.wrap_tool_call(request, handler)

    assert isinstance(result, Command)
    messages = result.update["messages"]
    assert isinstance(messages[0], ToolMessage)
    assert "attached below" in messages[0].content
    assert isinstance(messages[1], HumanMessage)


def test_media_middleware_wraps_pdf_payload(prompt_caller):
    middleware = prompt_caller._create_media_middleware()

    request = RequestStub("pdf-1")

    def handler(_request):
        return ResultStub(
            content=[
                {
                    "type": "input_file",
                    "filename": "model.pdf",
                    "file_data": "data:application/pdf;base64,AAA",
                }
            ]
        )

    result = middleware.wrap_tool_call(request, handler)

    assert isinstance(result, Command)
    assert "model.pdf" in result.update["messages"][0].content


def test_media_middleware_passthrough_for_non_media(prompt_caller):
    middleware = prompt_caller._create_media_middleware()

    original = ResultStub(content="plain text")

    def handler(_request):
        return original

    result = middleware.wrap_tool_call(RequestStub("plain-1"), handler)

    assert result is original


def test_media_middleware_parses_stringified_media_list(prompt_caller):
    middleware = prompt_caller._create_media_middleware()

    def handler(_request):
        return ResultStub(
            content="[{'type': 'input_file', 'filename': 'x.pdf', 'file_data': 'data:application/pdf;base64,AAA'}]"
        )

    result = middleware.wrap_tool_call(RequestStub("pdf-2"), handler)

    assert isinstance(result, Command)
    assert "x.pdf" in result.update["messages"][0].content
