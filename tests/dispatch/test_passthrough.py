"""Tests for the Anthropic passthrough module.

Uses `httpx.MockTransport` to intercept outbound requests so we can assert
URL, method, headers, and body all forward correctly without hitting the
real Anthropic API.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from magos.dispatch.passthrough import (
    call_passthrough,
    stream_passthrough,
)


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
        call_passthrough(
            raw_body,
            headers,
            "https://api.anthropic.com",
            path="/v1/messages",
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
        async for piece in stream_passthrough(
            raw_body,
            {"authorization": "Bearer x"},
            "https://api.anthropic.com",
            path="/v1/messages",
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
        async for piece in stream_passthrough(
            raw_body,
            {"authorization": "Bearer bad"},
            "https://api.anthropic.com",
            path="/v1/messages",
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
def test_call_passthrough_honours_path_parameter() -> None:
    """`path` is appended to `base_url`; lets the same module forward
    /v1/messages, /v1/responses, /v1/chat/completions, ..."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, content=b"{}", headers={"content-type": "application/json"})

    asyncio.run(
        call_passthrough(
            b"{}",
            {},
            "https://api.openai.com",
            path="/v1/responses",
            transport=httpx.MockTransport(handler),
        )
    )
    assert captured["url"] == "https://api.openai.com/v1/responses"


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
        call_passthrough(
            raw_body,
            {},
            "https://api.anthropic.com",
            path="/v1/messages",
            transport=httpx.MockTransport(handler),
        )
    )
    assert status == 429
