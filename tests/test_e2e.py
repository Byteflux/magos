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
