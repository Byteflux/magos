"""``/v1/chat/completions`` endpoint tests."""

from __future__ import annotations

import json as _json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from magos.ingress.http import create_app
from magos.routing import RoutingConfig

from ._helpers import OPENAI_CHAT_RESPONSE, client_with


@pytest.mark.integration
def test_chat_completions_endpoint_passes_through() -> None:
    openai_request = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
    }
    received: dict[str, Any] = {}

    async def fake_completion(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return OPENAI_CHAT_RESPONSE

    for client in client_with(chat_completion=fake_completion):
        resp = client.post("/v1/chat/completions", json=openai_request)

    assert resp.status_code == 200
    # Response model gets provider prefix added when request model lacks one
    expected_response = {**OPENAI_CHAT_RESPONSE, "model": f"openai/{openai_request['model']}"}
    assert resp.json() == expected_response
    expected = {**openai_request, "model": f"openai/{openai_request['model']}"}
    received_no_headers = {k: v for k, v in received.items() if k != "extra_headers"}
    assert received_no_headers == expected
    assert "extra_headers" in received


@pytest.mark.integration
def test_chat_completions_forwards_inbound_headers_to_dispatch() -> None:
    received: dict[str, Any] = {}

    async def fake_completion(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return OPENAI_CHAT_RESPONSE

    for client in client_with(chat_completion=fake_completion):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16,
            },
            headers={"Authorization": "Bearer key", "openai-organization": "org_123"},
        )

    assert resp.status_code == 200
    forwarded = received["extra_headers"]
    assert forwarded["authorization"] == "Bearer key"
    assert forwarded["openai-organization"] == "org_123"
    assert "content-type" not in {k.lower() for k in forwarded}


@pytest.mark.integration
def test_chat_completions_streams_sse() -> None:
    chunks = [
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4",
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4",
            "choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]

    received: dict[str, Any] = {}

    async def fake_iter() -> Any:
        for chunk in chunks:
            yield chunk

    async def fake_completion(**kwargs: Any) -> Any:
        received.update(kwargs)
        return fake_iter()

    body = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}], "stream": True}

    for client in client_with(chat_completion=fake_completion):
        with client.stream("POST", "/v1/chat/completions", json=body) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            text = b"".join(resp.iter_bytes()).decode()

    assert received["stream"] is True
    events = [line[len("data: ") :] for line in text.splitlines() if line.startswith("data: ")]
    assert len(events) == 4  # 3 chunks + [DONE]
    assert events[-1] == "[DONE]"
    parsed = [_json.loads(e) for e in events[:-1]]
    assert parsed[1]["choices"][0]["delta"]["content"] == "hello"
    assert parsed[2]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.unit
def test_unmatched_request_returns_404_with_openai_envelope() -> None:
    """Routing returns 404 with an OpenAI-shape error body for /v1/chat/completions."""
    cfg = RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"model": {"literal": "only-this-model"}},
                    "action": {"provider": "openai", "mode": "translate"},
                }
            ]
        }
    )
    body = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "x"}],
    }
    app = create_app(routing=cfg)
    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 404
    payload = resp.json()
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["code"] == "no_route_matched"
