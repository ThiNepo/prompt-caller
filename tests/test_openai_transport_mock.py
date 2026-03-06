import httpx
from openai import DefaultHttpxClient, OpenAI


def test_openai_client_can_run_with_mock_transport_only():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "id": "gpt-test",
                            "object": "model",
                            "created": 0,
                            "owned_by": "openai",
                        }
                    ],
                },
            )
        return httpx.Response(404, json={"error": {"message": "not found"}})

    client = OpenAI(
        api_key="test-key",
        base_url="https://example.test/v1",
        http_client=DefaultHttpxClient(transport=httpx.MockTransport(handler)),
    )

    models = client.models.list()
    assert models.data[0].id == "gpt-test"
