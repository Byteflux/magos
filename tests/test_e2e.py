"""End-to-end smoke tests against a real upstream provider.

Skipped by default. To run, set ``MAGOS_E2E=1`` and provide whatever upstream
credentials LiteLLM needs (e.g. ``OPENAI_API_KEY``)::

    MAGOS_E2E=1 OPENAI_API_KEY=... uv run pytest -m e2e

The model is configurable via ``MAGOS_E2E_MODEL`` (default ``gpt-4o-mini``);
any LiteLLM-supported model id works.

These tests exercise the full path: FastAPI -> magos.proxy -> litellm ->
real provider. They are intentionally minimal because each call costs money.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from magos.server import create_app

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("MAGOS_E2E") != "1",
        reason="set MAGOS_E2E=1 to run end-to-end provider tests",
    ),
]

MODEL = os.environ.get("MAGOS_E2E_MODEL", "gpt-4o-mini")
PROMPT = "Reply with the single word: pong"


def test_anthropic_non_streaming_real() -> None:
    body = {
        "model": MODEL,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": PROMPT}],
    }
    with TestClient(create_app()) as client:
        resp = client.post("/v1/messages", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert data["content"] and data["content"][0]["type"] == "text"


def test_anthropic_streaming_real() -> None:
    body = {
        "model": MODEL,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": True,
    }
    with (
        TestClient(create_app()) as client,
        client.stream("POST", "/v1/messages", json=body) as resp,
    ):
        assert resp.status_code == 200
        text = b"".join(resp.iter_bytes()).decode()
    assert "event: message_start" in text
    assert "event: message_stop" in text


def test_openai_non_streaming_real() -> None:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 16,
    }
    with TestClient(create_app()) as client:
        resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["role"] == "assistant"


def test_openai_streaming_real() -> None:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": True,
    }
    with (
        TestClient(create_app()) as client,
        client.stream("POST", "/v1/chat/completions", json=body) as resp,
    ):
        assert resp.status_code == 200
        text = b"".join(resp.iter_bytes()).decode()
    assert "data: [DONE]" in text


def test_anthropic_tool_use_round_trip_real() -> None:
    """Tool definition (Anthropic) -> tool_calls (OpenAI) -> tool_use (Anthropic)."""
    body = {
        "model": MODEL,
        "max_tokens": 64,
        "tools": [
            {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
        "tool_choice": {"type": "any"},
        "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
    }
    with TestClient(create_app()) as client:
        resp = client.post("/v1/messages", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["stop_reason"] == "tool_use"
    tool_uses = [b for b in data["content"] if b.get("type") == "tool_use"]
    assert tool_uses, f"expected tool_use block, got {data['content']}"
    assert tool_uses[0]["name"] == "get_weather"
    assert "city" in tool_uses[0]["input"]


def test_anthropic_multi_turn_with_system_real() -> None:
    """System prompt + multi-turn history survive translation."""
    body = {
        "model": MODEL,
        "max_tokens": 16,
        "system": "You always reply with exactly one word, lowercase, no punctuation.",
        "messages": [
            {"role": "user", "content": "Say hello."},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "Now say goodbye."},
        ],
    }
    with TestClient(create_app()) as client:
        resp = client.post("/v1/messages", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    text_blocks = [b for b in data["content"] if b.get("type") == "text"]
    assert text_blocks, data
    reply = text_blocks[0]["text"].strip().lower()
    assert "goodbye" in reply or "bye" in reply, f"expected farewell, got {reply!r}"


def test_anthropic_count_tokens_real() -> None:
    """count_tokens returns a positive estimate for an OpenAI model (local path)."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hello world, count me."}],
    }
    with TestClient(create_app()) as client:
        resp = client.post("/v1/messages/count_tokens", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert isinstance(data["input_tokens"], int)
    assert data["input_tokens"] > 0
