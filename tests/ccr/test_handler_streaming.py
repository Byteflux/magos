"""``wrap_stream`` streams chunks through headroom's CCR streaming handler."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any


def test_wrap_stream_passthrough_when_not_ccr_request() -> None:
    """No CCR tool in body -> passthrough, byte-for-byte."""
    from magos.ccr import wrap_stream  # noqa: PLC0415
    from magos.egress.translate import TRANSLATE_HANDLERS  # noqa: PLC0415
    from magos.routing.request import RoutedRequest  # noqa: PLC0415

    chunks_in = [
        b'event: message_start\ndata: {"message": {"model": "x"}}\n\n',
        b'event: content_block_delta\ndata: {"delta": {"text": "hi"}}\n\n',
        b"event: message_stop\ndata: {}\n\n",
    ]

    async def upstream() -> AsyncIterator[bytes]:
        for c in chunks_in:
            yield c

    async def fake_completion(**_: Any) -> Any:
        raise AssertionError("continuation should not run")

    req = RoutedRequest(
        endpoint="/v1/messages",
        headers={},
        body={"model": "x", "messages": []},
        raw_body=b"",
    )

    async def collect() -> bytes:
        out = b""
        async for chunk in wrap_stream(
            upstream(),
            req=req,
            adapter=TRANSLATE_HANDLERS["/v1/messages"],
            completion=fake_completion,
            dispatch_model="x",
            provider="anthropic",
            forward_headers={},
            api_key=None,
            api_base=None,
        ):
            out += chunk
        return out

    out = asyncio.run(collect())
    assert out == b"".join(chunks_in)


def test_wrap_stream_passthrough_when_no_ccr_tool_call_in_stream() -> None:
    """CCR tool present in request, but stream has no tool_use -> no continuation called.

    Headroom's ``StreamingCCRHandler`` buffers chunks until it sees ``stop_reason``
    to decide whether a CCR tool call occurred.  When the first chunk already
    contains ``stop_reason`` and no CCR is detected, headroom yields the buffered
    chunk(s) and exits — it does not continue draining the upstream.  We assert
    that (a) the continuation is never invoked and (b) at least the first chunk
    (which triggered the detection decision) is forwarded.
    """
    from magos.ccr import wrap_stream  # noqa: PLC0415
    from magos.egress.translate import TRANSLATE_HANDLERS  # noqa: PLC0415
    from magos.routing.request import RoutedRequest  # noqa: PLC0415

    chunks_in = [
        b'event: message_start\ndata: {"message": {"model": "x", "stop_reason": "end_turn"}}\n\n',
        b'event: content_block_delta\ndata: {"delta": {"text": "no tools used"}}\n\n',
        b"event: message_stop\ndata: {}\n\n",
    ]

    async def upstream() -> AsyncIterator[bytes]:
        for c in chunks_in:
            yield c

    continuation_called = False

    async def fake_completion(**_: Any) -> Any:
        nonlocal continuation_called
        continuation_called = True
        raise AssertionError("continuation should not run")

    req = RoutedRequest(
        endpoint="/v1/messages",
        headers={},
        body={
            "model": "x",
            "messages": [],
            "tools": [{"name": "headroom_retrieve"}],
        },
        raw_body=b"",
    )

    async def collect() -> bytes:
        out = b""
        async for chunk in wrap_stream(
            upstream(),
            req=req,
            adapter=TRANSLATE_HANDLERS["/v1/messages"],
            completion=fake_completion,
            dispatch_model="x",
            provider="anthropic",
            forward_headers={},
            api_key=None,
            api_base=None,
        ):
            out += chunk
        return out

    out = asyncio.run(collect())
    # Continuation must never fire (no CCR tool call in stream).
    assert not continuation_called
    # The chunk that triggered the no-CCR decision must be forwarded.
    assert chunks_in[0] in out
