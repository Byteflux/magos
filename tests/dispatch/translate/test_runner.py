"""Unit tests for the generic translate runner (proxy_translate / stream_translate).

These tests verify the ``on_complete`` kwarg is accepted and forwarded to
``log_usage_from_body`` / ``tap_stream``.  The functional behaviour of each
per-shape adapter is covered by the sibling test files.
"""

from __future__ import annotations

import asyncio
from typing import Any

from magos.dispatch.translate import TRANSLATE_HANDLERS
from magos.dispatch.translate.runner import proxy_translate, stream_translate
from magos.dispatch.usage import Usage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ANTHROPIC_ADAPTER = TRANSLATE_HANDLERS["/v1/messages"]

_MINIMAL_REQUEST = {
    "model": "claude-x",
    "max_tokens": 4,
    "messages": [{"role": "user", "content": "hi"}],
}

_ANTHROPIC_RESPONSE_WITH_USAGE: dict[str, Any] = {
    "type": "message",
    "id": "msg_1",
    "role": "assistant",
    "content": [{"type": "text", "text": "ok"}],
    "model": "claude-x",
    "usage": {"input_tokens": 100, "output_tokens": 20},
}


# ---------------------------------------------------------------------------
# proxy_translate
# ---------------------------------------------------------------------------


def test_proxy_translate_on_complete_fires_with_usage() -> None:
    """``on_complete`` is called once with the extracted ``Usage`` on success."""

    seen: list[Usage] = []

    async def fake(**_: Any) -> dict[str, Any]:
        return _ANTHROPIC_RESPONSE_WITH_USAGE

    asyncio.run(
        proxy_translate(
            _ANTHROPIC_ADAPTER,
            _MINIMAL_REQUEST,
            dispatch_model="anthropic/claude-x",
            completion=fake,
            on_complete=seen.append,
        )
    )

    assert len(seen) == 1
    assert seen[0].input == 100
    assert seen[0].output == 20


def test_proxy_translate_on_complete_not_called_when_usage_empty() -> None:
    """``on_complete`` is NOT called when the response carries no usage."""

    seen: list[Usage] = []

    async def fake(**_: Any) -> dict[str, Any]:
        # No ``usage`` key → Usage() → is_empty == True
        return {
            "type": "message",
            "id": "msg_2",
            "role": "assistant",
            "content": [],
            "model": "claude-x",
        }

    asyncio.run(
        proxy_translate(
            _ANTHROPIC_ADAPTER,
            _MINIMAL_REQUEST,
            dispatch_model="anthropic/claude-x",
            completion=fake,
            on_complete=seen.append,
        )
    )

    assert seen == []


def test_proxy_translate_without_on_complete_is_unaffected() -> None:
    """Omitting ``on_complete`` (default ``None``) keeps existing behaviour."""

    async def fake(**_: Any) -> dict[str, Any]:
        return _ANTHROPIC_RESPONSE_WITH_USAGE

    result = asyncio.run(
        proxy_translate(
            _ANTHROPIC_ADAPTER,
            _MINIMAL_REQUEST,
            dispatch_model="anthropic/claude-x",
            completion=fake,
        )
    )

    assert result["type"] == "message"


# ---------------------------------------------------------------------------
# stream_translate
# ---------------------------------------------------------------------------


def test_stream_translate_accepts_on_complete_kwarg() -> None:
    """``stream_translate`` accepts ``on_complete`` and does not raise."""

    chunks = [
        b'event: message_start\ndata: {"type": "message_start", "message": {"usage": {"input_tokens": 50, "output_tokens": 0}, "model": "claude-x"}}\n\n',
        b'event: message_delta\ndata: {"type": "message_delta", "usage": {"output_tokens": 10}}\n\n',
        b'event: message_stop\ndata: {"type": "message_stop"}\n\n',
    ]

    seen: list[Usage] = []

    async def fake_iter() -> Any:
        for chunk in chunks:
            yield chunk

    async def fake(**_: Any) -> Any:
        return fake_iter()

    async def collect() -> list[bytes]:
        return [
            chunk
            async for chunk in stream_translate(
                _ANTHROPIC_ADAPTER,
                _MINIMAL_REQUEST,
                dispatch_model="anthropic/claude-x",
                completion=fake,
                on_complete=seen.append,
            )
        ]

    received = asyncio.run(collect())
    assert received == chunks
    # on_complete fires once the stream is fully consumed and usage is non-empty.
    assert len(seen) == 1
    assert seen[0].input == 50
    assert seen[0].output == 10


def test_stream_translate_without_on_complete_is_unaffected() -> None:
    """Omitting ``on_complete`` keeps existing streaming behaviour."""

    chunks = [b"event: message_stop\ndata: {}\n\n"]

    async def fake_iter() -> Any:
        for chunk in chunks:
            yield chunk

    async def fake(**_: Any) -> Any:
        return fake_iter()

    async def collect() -> list[bytes]:
        return [
            chunk
            async for chunk in stream_translate(
                _ANTHROPIC_ADAPTER,
                _MINIMAL_REQUEST,
                dispatch_model="anthropic/claude-x",
                completion=fake,
            )
        ]

    received = asyncio.run(collect())
    assert received == chunks
