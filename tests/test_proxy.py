"""Unit tests for the LiteLLM-backed proxy entry points.

After the LiteLLM SDK fold-in there is no per-field translation to verify;
``proxy_anthropic_messages`` is a thin marshalling layer over
``litellm.anthropic_messages``. These tests check that contract: payload
composition (model rewrite, header forwarding, api_key threading) and
response coercion. The cross-provider behavior itself is covered by the
e2e suite (``MAGOS_E2E=1``).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from magos.proxy import proxy_anthropic_messages, stream_anthropic_messages


@pytest.mark.unit
def test_proxy_anthropic_messages_passes_dispatch_model_and_returns_dict() -> None:
    received: dict[str, Any] = {}

    async def fake(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
        }

    request = {
        "model": "claude-haiku",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    }
    out = asyncio.run(
        proxy_anthropic_messages(
            request,
            dispatch_model="anthropic/claude-haiku-4-5",
            completion=fake,
        )
    )
    # dispatch_model overrides the inbound body's model.
    assert received["model"] == "anthropic/claude-haiku-4-5"
    assert received["max_tokens"] == 16
    assert out["type"] == "message"
    assert out["content"][0]["text"] == "hi"


@pytest.mark.unit
def test_proxy_anthropic_messages_coerces_pydantic_like_response() -> None:
    class _PydanticLike:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def model_dump(self) -> dict[str, Any]:
            return self._payload

    async def fake(**_: Any) -> _PydanticLike:
        return _PydanticLike({"type": "message", "role": "assistant", "content": []})

    out = asyncio.run(
        proxy_anthropic_messages(
            {
                "model": "x",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "x"}],
            },
            dispatch_model="anthropic/x",
            completion=fake,
        )
    )
    assert out["type"] == "message"


@pytest.mark.unit
def test_proxy_anthropic_messages_threads_api_key_and_headers() -> None:
    received: dict[str, Any] = {}

    async def fake(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return {"type": "message", "role": "assistant", "content": []}

    asyncio.run(
        proxy_anthropic_messages(
            {
                "model": "x",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "x"}],
            },
            dispatch_model="anthropic/x",
            completion=fake,
            forward_headers={
                "authorization": "Bearer xyz",
                "anthropic-beta": "feature-x",
                # Hop-by-hop / SDK-owned headers must be filtered.
                "content-type": "application/json",
                "content-length": "123",
            },
            api_key="explicit-key",
        )
    )
    assert received["api_key"] == "explicit-key"
    forwarded = received["extra_headers"]
    assert forwarded["authorization"] == "Bearer xyz"
    assert forwarded["anthropic-beta"] == "feature-x"
    assert "content-type" not in forwarded
    assert "content-length" not in forwarded


@pytest.mark.unit
def test_stream_anthropic_messages_forwards_bytes_verbatim() -> None:
    chunks = [
        b'event: message_start\ndata: {"type": "message_start"}\n\n',
        b'event: content_block_delta\ndata: {"type": "content_block_delta"}\n\n',
        b'event: message_stop\ndata: {"type": "message_stop"}\n\n',
    ]

    async def fake_iter() -> Any:
        for chunk in chunks:
            yield chunk

    async def fake(**_: Any) -> Any:
        return fake_iter()

    request = {
        "model": "x",
        "max_tokens": 4,
        "messages": [{"role": "user", "content": "hi"}],
    }
    out = stream_anthropic_messages(request, dispatch_model="anthropic/x", completion=fake)

    async def collect() -> list[bytes]:
        return [chunk async for chunk in out]

    received = asyncio.run(collect())
    assert received == chunks


@pytest.mark.unit
def test_stream_anthropic_messages_forces_stream_true() -> None:
    received: dict[str, Any] = {}

    async def fake_iter() -> Any:
        for chunk in (b"event: x\ndata: {}\n\n",):
            yield chunk

    async def fake(**kwargs: Any) -> Any:
        received.update(kwargs)
        return fake_iter()

    request = {
        "model": "x",
        "max_tokens": 4,
        "messages": [{"role": "user", "content": "hi"}],
    }

    async def drain() -> None:
        async for _ in stream_anthropic_messages(
            request, dispatch_model="anthropic/x", completion=fake
        ):
            pass

    asyncio.run(drain())
    assert received["stream"] is True


@pytest.mark.unit
def test_stream_anthropic_messages_emits_error_event_on_dispatch_failure() -> None:
    async def boom(**_: Any) -> Any:
        raise RuntimeError("upstream exploded")

    request = {
        "model": "x",
        "max_tokens": 4,
        "messages": [{"role": "user", "content": "hi"}],
    }
    out = stream_anthropic_messages(request, dispatch_model="anthropic/x", completion=boom)

    async def collect() -> list[bytes]:
        return [chunk async for chunk in out]

    received = asyncio.run(collect())
    assert len(received) == 1
    text = received[0].decode()
    assert "event: error" in text
    assert "upstream exploded" in text
