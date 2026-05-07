"""``tap_stream``: forwards SSE bytes verbatim while accumulating usage stats."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

from magos.shapes import Shape

from .accumulator import UsageAccumulator
from .core import Usage, log_usage


def _iter_complete_events(buf: bytes) -> tuple[list[bytes], bytes]:
    """Split ``buf`` on event boundaries (``\\n\\n``); return (events, leftover)."""
    parts = buf.split(b"\n\n")
    if len(parts) == 1:
        return [], buf
    return parts[:-1], parts[-1]


def _parse_event(raw: bytes) -> tuple[str | None, dict[str, Any] | None]:
    """Parse one SSE event into ``(event_name, data_object)``.

    Multiple ``data:`` lines per event are joined with newlines (SSE spec).
    Non-JSON payloads (e.g. ``[DONE]``) yield ``None`` for the data dict.
    """
    name: str | None = None
    data_lines: list[str] = []
    for line in raw.split(b"\n"):
        decoded = line.decode("utf-8", errors="replace")
        if decoded.startswith("event:"):
            name = decoded[len("event:") :].strip()
        elif decoded.startswith("data:"):
            data_lines.append(decoded[len("data:") :].lstrip())
    if not data_lines:
        return name, None
    raw_data = "\n".join(data_lines)
    try:
        parsed = json.loads(raw_data)
    except (ValueError, TypeError):
        return name, None
    return name, parsed if isinstance(parsed, dict) else None


async def tap_stream(
    upstream: AsyncIterator[bytes],
    shape: Shape,
    *,
    endpoint: str,
    fallback_model: str | None = None,
    on_complete: Callable[[Usage], None] | None = None,
) -> AsyncIterator[bytes]:
    """Forward ``upstream`` byte-for-byte while accumulating usage stats.

    Usage parsing is best-effort: malformed/truncated streams or upstreams
    that omit a final usage block degrade silently to no log.

    If ``on_complete`` is provided and the final accumulated usage is
    non-empty, it is invoked once after final logging, even if the stream
    raised mid-way.
    """
    accumulator = UsageAccumulator(shape)
    buf = b""
    try:
        async for chunk in upstream:
            buf += chunk
            events, buf = _iter_complete_events(buf)
            for event_bytes in events:
                event_name, data = _parse_event(event_bytes)
                if data is not None:
                    accumulator.feed(event_name, data)
            yield chunk
    finally:
        # Flush any final event held in the buffer (some upstreams omit
        # the trailing blank line on the last event).
        if buf.strip():
            event_name, data = _parse_event(buf)
            if data is not None:
                accumulator.feed(event_name, data)
        snapshot = accumulator.snapshot()
        log_usage(
            shape,
            endpoint=endpoint,
            model=accumulator.model or fallback_model,
            usage=snapshot,
            stream=True,
        )
        if on_complete is not None and not snapshot.is_empty:
            on_complete(snapshot)
