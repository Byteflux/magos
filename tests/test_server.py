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
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from magos import tokens
from magos.config import MagosSettings, get_settings
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


@pytest.mark.integration
def test_messages_streams_anthropic_events() -> None:
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
            "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]

    async def fake_iter() -> Any:
        for chunk in chunks:
            yield chunk

    async def fake_completion(**_: Any) -> Any:
        return fake_iter()

    app = create_app()
    body = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }

    for client in _client_with(app, fake_completion):
        with client.stream("POST", "/v1/messages", json=body) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            text = b"".join(resp.iter_bytes()).decode()

    event_types = [
        line[len("event: ") :] for line in text.splitlines() if line.startswith("event: ")
    ]
    assert event_types == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    data_lines = [line[len("data: ") :] for line in text.splitlines() if line.startswith("data: ")]
    parsed = [json.loads(line) for line in data_lines]
    assert parsed[2]["delta"] == {"type": "text_delta", "text": "hi"}
    assert parsed[4]["delta"]["stop_reason"] == "end_turn"
    # message_start.usage.input_tokens is seeded by the local LiteLLM
    # estimator, so it should be a positive integer for a non-empty request.
    assert parsed[0]["message"]["usage"]["input_tokens"] > 0


@pytest.mark.unit
def test_messages_streaming_returns_400_on_invalid_request() -> None:
    app = create_app()
    body = {"model": "x", "stream": True}  # missing required fields
    with TestClient(app) as client:
        resp = client.post("/v1/messages", json=body)
    assert resp.status_code == 400


@pytest.mark.integration
def test_count_tokens_endpoint_local_path() -> None:
    """Empty passthrough set forces the local estimator."""
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: MagosSettings(
        count_tokens_passthrough_providers=frozenset(),
        _env_file=None,  # type: ignore[call-arg]
    )
    body = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hello there"}],
    }
    try:
        with TestClient(app) as client:
            resp = client.post("/v1/messages/count_tokens", json=body)
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 200
    payload = resp.json()
    assert isinstance(payload["input_tokens"], int)
    assert payload["input_tokens"] > 0


@pytest.mark.integration
def test_count_tokens_endpoint_uses_passthrough_when_allowed() -> None:
    """anthropic in allow-list + claude- model -> patched passthrough is hit."""
    captured: dict[str, Any] = {}

    async def fake_passthrough(req: dict[str, Any]) -> int:
        captured["model"] = req["model"]
        return 4242

    app = create_app()
    app.dependency_overrides[get_settings] = lambda: MagosSettings(
        count_tokens_passthrough_providers=frozenset({"anthropic"}),
        _env_file=None,  # type: ignore[call-arg]
    )
    body = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    }
    try:
        with (
            patch.dict(tokens.PASSTHROUGH_DISPATCH, {"anthropic": fake_passthrough}),
            TestClient(app) as client,
        ):
            resp = client.post("/v1/messages/count_tokens", json=body)
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert resp.json() == {"input_tokens": 4242}
    assert captured["model"] == "claude-3-5-sonnet-20241022"


@pytest.mark.unit
def test_count_tokens_endpoint_returns_400_on_invalid_request() -> None:
    app = create_app()
    with TestClient(app) as client:
        resp = client.post("/v1/messages/count_tokens", json={"model": "x"})
    assert resp.status_code == 400


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
