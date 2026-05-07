"""Unit tests for `magos.dispatch.translate.safe_stream`.

Verify the wrapper:
  * forwards chunks transparently when no exception occurs;
  * on a mid-stream exception, emits a structured `egress.stream_error`
    log and a per-shape SSE error frame so the client can unwind.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from magos.dispatch.translate import safe_stream as safe_stream_mod
from magos.dispatch.translate.safe_stream import safe_stream
from magos.shapes import ANTHROPIC, OPENAI_CHAT, OPENAI_RESPONSES, Shape


async def _gen(*chunks: bytes) -> AsyncIterator[bytes]:
    for c in chunks:
        yield c


async def _raising(*chunks: bytes, exc: BaseException) -> AsyncIterator[bytes]:
    for c in chunks:
        yield c
    raise exc


async def _drain(it: AsyncIterator[bytes]) -> list[bytes]:
    return [c async for c in it]


@pytest.mark.unit
def test_passthrough_on_success() -> None:
    async def go() -> list[bytes]:
        return await _drain(
            safe_stream(
                _gen(b"a", b"b"),
                shape=ANTHROPIC,
                endpoint="/v1/messages",
                model="claude-x",
            )
        )

    assert asyncio.run(go()) == [b"a", b"b"]


@pytest.mark.unit
def test_anthropic_emits_named_error_event() -> None:
    async def go() -> list[bytes]:
        return await _drain(
            safe_stream(
                _raising(b"chunk", exc=RuntimeError("upstream 500")),
                shape=ANTHROPIC,
                endpoint="/v1/messages",
                model="claude-x",
            )
        )

    out = asyncio.run(go())
    assert out[0] == b"chunk"
    last = out[-1].decode()
    assert last.startswith("event: error\n")
    assert '"type": "error"' in last
    assert "upstream 500" in last


@pytest.mark.unit
def test_openai_chat_emits_data_error_then_done() -> None:
    async def go() -> list[bytes]:
        return await _drain(
            safe_stream(
                _raising(exc=RuntimeError("upstream 500")),
                shape=OPENAI_CHAT,
                endpoint="/v1/chat/completions",
                model="gpt-x",
            )
        )

    out = asyncio.run(go())
    last = out[-1].decode()
    # Chat has no `event:` line; the convention is a `data: {error}` frame
    # followed by `data: [DONE]` so the client unwinds cleanly.
    assert last.startswith("data: ")
    assert '"error"' in last
    assert "data: [DONE]" in last


@pytest.mark.unit
def test_openai_responses_emits_named_error_event() -> None:
    async def go() -> list[bytes]:
        return await _drain(
            safe_stream(
                _raising(exc=RuntimeError("upstream 500")),
                shape=OPENAI_RESPONSES,
                endpoint="/v1/responses",
                model="gpt-x",
            )
        )

    out = asyncio.run(go())
    last = out[-1].decode()
    assert last.startswith("event: error\n")
    assert '"type": "error"' in last


@pytest.mark.unit
def test_unknown_shape_logs_but_emits_no_frame() -> None:
    """Defensive: an unrecognised shape name still logs cleanly; no SSE frame."""
    custom = Shape(
        name="custom",
        endpoints=("/v1/custom",),
        compression_provider="anthropic",
        system_field=None,
        messages_field=None,
        instructions_field=None,
        usage_keys={},
        stream_events=(),
    )

    async def go() -> list[bytes]:
        return await _drain(
            safe_stream(
                _raising(b"x", exc=RuntimeError("boom")),
                shape=custom,
                endpoint="/v1/custom",
                model="m",
            )
        )

    assert asyncio.run(go()) == [b"x"]


@pytest.mark.unit
def test_emits_structured_log_event(monkeypatch: pytest.MonkeyPatch) -> None:
    """The structured log carries shape/endpoint/model/error_type for alerting."""
    captured: list[dict[str, Any]] = []

    class _Recorder:
        def warning(self, event: str, **kwargs: Any) -> None:
            captured.append({"event": event, **kwargs})

    monkeypatch.setattr(safe_stream_mod, "log", _Recorder())

    async def go() -> None:
        await _drain(
            safe_stream(
                _raising(exc=ValueError("nope")),
                shape=ANTHROPIC,
                endpoint="/v1/messages",
                model="claude-x",
            )
        )

    asyncio.run(go())

    assert len(captured) == 1
    e = captured[0]
    assert e == {
        "event": "egress.stream_error",
        "shape": "anthropic",
        "endpoint": "/v1/messages",
        "model": "claude-x",
        "error": "nope",
        "error_type": "ValueError",
    }
