"""FastAPI endpoint tests for the magos server.

Drives the server with TestClient and overrides the completion dependency so
no real upstream is contacted. Covers both endpoints, basic error paths, and
the streaming-not-yet-implemented gate.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from magos.server import create_app, get_completion

FIXTURES_ROOT = Path(__file__).parent / "fixtures" / "translation"


def _load(case_dir: Path, name: str) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((case_dir / name).read_text(encoding="utf-8")))


def _client_with(app: FastAPI, completion: Any) -> Iterator[TestClient]:
    app.dependency_overrides[get_completion] = lambda: completion
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


@pytest.mark.integration
def test_messages_endpoint_round_trip() -> None:
    case_dir = FIXTURES_ROOT / "simple_text"
    anthropic_request = _load(case_dir, "anthropic_request.json")
    expected_openai_request = _load(case_dir, "openai_request.json")
    openai_response = _load(case_dir, "openai_response.json")
    expected_anthropic_response = _load(case_dir, "anthropic_response.json")

    received: dict[str, Any] = {}

    async def fake_completion(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return openai_response

    app = create_app()
    for client in _client_with(app, fake_completion):
        resp = client.post("/v1/messages", json=anthropic_request)

    assert resp.status_code == 200
    body = resp.json()
    assert received == expected_openai_request

    expected_no_id = {k: v for k, v in expected_anthropic_response.items() if k != "id"}
    body_no_id = {k: v for k, v in body.items() if k != "id"}
    assert body_no_id == expected_no_id


@pytest.mark.integration
def test_chat_completions_endpoint_passes_through() -> None:
    openai_request = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
    }
    openai_response = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    received: dict[str, Any] = {}

    async def fake_completion(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return openai_response

    app = create_app()
    for client in _client_with(app, fake_completion):
        resp = client.post("/v1/chat/completions", json=openai_request)

    assert resp.status_code == 200
    assert resp.json() == openai_response
    assert received == openai_request


@pytest.mark.unit
def test_messages_rejects_streaming() -> None:
    app = create_app()
    body = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    with TestClient(app) as client:
        resp = client.post("/v1/messages", json=body)
    assert resp.status_code == 501
    assert resp.json()["detail"] == "streaming not yet implemented"


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

    app = create_app()
    body = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}], "stream": True}

    for client in _client_with(app, fake_completion):
        with client.stream("POST", "/v1/chat/completions", json=body) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            text = b"".join(resp.iter_bytes()).decode()

    assert received["stream"] is True
    events = [line[len("data: ") :] for line in text.splitlines() if line.startswith("data: ")]
    assert len(events) == 4  # 3 chunks + [DONE]
    assert events[-1] == "[DONE]"
    parsed = [json.loads(e) for e in events[:-1]]
    assert parsed[1]["choices"][0]["delta"]["content"] == "hello"
    assert parsed[2]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.unit
def test_messages_returns_400_on_invalid_request() -> None:
    app = create_app()
    with TestClient(app) as client:
        resp = client.post("/v1/messages", json={"model": "x"})  # missing required fields
    assert resp.status_code == 400


@pytest.mark.unit
def test_messages_returns_502_on_upstream_failure() -> None:
    case_dir = FIXTURES_ROOT / "simple_text"
    anthropic_request = _load(case_dir, "anthropic_request.json")

    async def boom(**_: Any) -> dict[str, Any]:
        raise RuntimeError("upstream exploded")

    app = create_app()
    for client in _client_with(app, boom):
        resp = client.post("/v1/messages", json=anthropic_request)

    assert resp.status_code == 502
    assert "upstream exploded" in resp.json()["detail"]
