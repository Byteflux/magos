"""Tests for the Anthropic passthrough module.

Uses ``httpx.MockTransport`` to intercept outbound requests so we can assert
URL, method, headers, and body all forward correctly without hitting the
real Anthropic API.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from magos.passthrough import (
    call_anthropic_passthrough,
    should_anthropic_passthrough,
    stream_anthropic_passthrough,
)


@pytest.mark.unit
def test_should_anthropic_passthrough_for_claude_model() -> None:
    assert should_anthropic_passthrough({"model": "claude-sonnet-4-6"}) is True
    assert should_anthropic_passthrough({"model": "claude-3-5-sonnet-20241022"}) is True


@pytest.mark.unit
def test_should_anthropic_passthrough_false_for_others() -> None:
    assert should_anthropic_passthrough({"model": "gpt-4"}) is False
    assert should_anthropic_passthrough({"model": "unknown-model"}) is False
    assert should_anthropic_passthrough({}) is False


@pytest.mark.unit
def test_call_passthrough_forwards_request_verbatim() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=b'{"id":"msg_1","type":"message","role":"assistant"}',
            headers={"content-type": "application/json"},
        )

    body = {"model": "claude-sonnet-4-6", "max_tokens": 16, "messages": []}
    raw_body = json.dumps(body).encode()
    headers = {"authorization": "Bearer test-oauth", "anthropic-beta": "x,y"}

    status, raw, ct = asyncio.run(
        call_anthropic_passthrough(
            raw_body,
            headers,
            "https://api.anthropic.com",
            transport=httpx.MockTransport(handler),
        )
    )

    assert status == 200
    assert ct == "application/json"
    assert b'"id":"msg_1"' in raw
    assert captured["method"] == "POST"
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["body"] == body
    assert captured["headers"]["authorization"] == "Bearer test-oauth"
    assert captured["headers"]["anthropic-beta"] == "x,y"


@pytest.mark.unit
def test_stream_passthrough_yields_upstream_bytes_verbatim() -> None:
    chunks = [
        b'event: message_start\ndata: {"type":"message_start"}\n\n',
        b'event: content_block_delta\ndata: {"type":"content_block_delta"}\n\n',
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"".join(chunks),
            headers={"content-type": "text/event-stream"},
        )

    body = {"model": "claude-sonnet-4-6", "max_tokens": 16, "messages": [], "stream": True}
    raw_body = json.dumps(body).encode()

    async def run() -> bytes:
        out = b""
        async for piece in stream_anthropic_passthrough(
            raw_body,
            {"authorization": "Bearer x"},
            "https://api.anthropic.com",
            transport=httpx.MockTransport(handler),
        ):
            out += piece
        return out

    result = asyncio.run(run())
    # Bytes are forwarded verbatim, so the concatenated output equals the
    # concatenated upstream chunks (httpx may rebuffer, but the content is
    # identical).
    assert result == b"".join(chunks)


@pytest.mark.unit
def test_stream_passthrough_emits_error_event_on_upstream_4xx() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            content=b'{"type":"error","error":{"type":"authentication_error"}}',
            headers={"content-type": "application/json"},
        )

    body = {"model": "claude-sonnet-4-6", "max_tokens": 16, "messages": [], "stream": True}
    raw_body = json.dumps(body).encode()

    async def run() -> bytes:
        out = b""
        async for piece in stream_anthropic_passthrough(
            raw_body,
            {"authorization": "Bearer bad"},
            "https://api.anthropic.com",
            transport=httpx.MockTransport(handler),
        ):
            out += piece
        return out

    result = asyncio.run(run())
    text = result.decode()
    assert "event: error" in text
    assert '"status":401' in text
    assert "authentication_error" in text


@pytest.mark.unit
def test_call_passthrough_propagates_upstream_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            content=b'{"type":"error"}',
            headers={"content-type": "application/json"},
        )

    raw_body = json.dumps({"model": "claude-sonnet-4-6", "max_tokens": 16, "messages": []}).encode()
    status, _, _ = asyncio.run(
        call_anthropic_passthrough(
            raw_body,
            {},
            "https://api.anthropic.com",
            transport=httpx.MockTransport(handler),
        )
    )
    assert status == 429
