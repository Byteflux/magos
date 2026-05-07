"""End-to-end byte-exact passthrough tests for /v1/messages with OAuth headers.

These exercise the shipped routing config's claude-* passthrough rule
with the Claude-Code-style header set (OAuth bearer + ``anthropic-beta``
+ ``anthropic-version``). See ``tests/e2e/conftest.py`` for the
``MAGOS_E2E=1`` skip gate.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from magos.api import build_api

from ._helpers import (
    ANTHROPIC_MODEL,
    PROMPT,
    anthropic_inbound_headers,
    maybe_skip_anthropic_oauth,
)


def test_anthropic_passthrough_with_oauth_headers_real() -> None:
    """Byte-exact passthrough of /v1/messages with Claude-Code-style headers.

    Validates that magos forwards them verbatim and the upstream accepts
    them (subject to Extra Usage being enabled on the subscription).
    """
    maybe_skip_anthropic_oauth()
    body = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": PROMPT}],
    }
    with TestClient(build_api()) as client:
        resp = client.post("/v1/messages", json=body, headers=anthropic_inbound_headers())
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert data["content"] and data["content"][0]["type"] == "text"


def test_anthropic_passthrough_streaming_with_oauth_headers_real() -> None:
    """Byte-exact passthrough streaming with Claude-Code-style headers.

    Same as the non-streaming sibling but verifies that
    ``passthrough.stream_passthrough`` round-trips the SSE bytes verbatim.
    """
    maybe_skip_anthropic_oauth()
    body = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": True,
    }
    with (
        TestClient(build_api()) as client,
        client.stream(
            "POST", "/v1/messages", json=body, headers=anthropic_inbound_headers()
        ) as resp,
    ):
        assert resp.status_code == 200
        text = b"".join(resp.iter_bytes()).decode()
    assert "event: message_start" in text
    assert "event: message_stop" in text
