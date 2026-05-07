"""SSE frame helpers."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any


def sse_event(data: str) -> bytes:
    return f"data: {data}\n\n".encode()


def sse_named_event(event: dict[str, Any]) -> bytes:
    """OpenAI Responses streaming uses ``event:`` + ``data:`` lines per chunk."""
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


async def rewrite_data_in_stream(
    upstream: AsyncIterator[bytes],
    mutator: Callable[[dict[str, Any]], bool],
) -> AsyncIterator[bytes]:
    """Forward ``upstream`` chunks, re-emitting any ``data:`` JSON the mutator updates.

    ``mutator(data)`` returns True iff it changed ``data`` and the chunk should be
    re-serialized; otherwise the chunk is forwarded verbatim. Chunks may carry an
    ``event:`` line before ``data:`` (Anthropic, OpenAI Responses); the prefix is
    preserved on re-emit. Non-parseable chunks, ``[DONE]``, and chunks without a
    ``data:`` line pass through unchanged.
    """
    needle = b'"model"'
    async for chunk in upstream:
        if needle not in chunk:
            yield chunk
            continue
        try:
            text = chunk.decode("utf-8")
        except UnicodeDecodeError:
            yield chunk
            continue
        prefix, sep, after = text.partition("data:")
        if not sep:
            yield chunk
            continue
        data_str = after.strip()
        if data_str == "[DONE]":
            yield chunk
            continue
        try:
            data = json.loads(data_str)
        except ValueError:
            yield chunk
            continue
        if not isinstance(data, dict) or not mutator(data):
            yield chunk
            continue
        yield f"{prefix}data: {json.dumps(data)}\n\n".encode()
