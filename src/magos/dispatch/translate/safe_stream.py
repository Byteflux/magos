"""Convert mid-stream exceptions into a structured log + per-shape SSE error.

A streaming dispatch returns an `AsyncIterator[bytes]` that's consumed by
Starlette long after `Gateway.dispatch` has returned. An exception raised
during iteration (e.g. LiteLLM's `MidStreamFallbackError` when an upstream
provider hands back a 5xx mid-stream) bubbles past the service-layer
`try/except` in `magos.service.request` and lands in stdlib logging via
Starlette/uvicorn — where our `ConsoleRenderer` then renders it as a Rich
traceback dumping every frame's locals (request payload, headers, SDK state).

`safe_stream` wraps the iterator at the gateway boundary so the exception
becomes a single structured warning and the client receives a graceful
end-of-stream with a per-shape SSE error event.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from magos.shapes import Shape
from magos.telemetry import get_logger

log = get_logger("magos.dispatch.translate")


def _anthropic_error_frame(message: str) -> bytes:
    """Anthropic Messages SSE error event (named event, single frame)."""
    payload = json.dumps({"type": "error", "error": {"type": "api_error", "message": message}})
    return f"event: error\ndata: {payload}\n\n".encode()


def _openai_chat_error_frame(message: str) -> bytes:
    """OpenAI Chat has no formal SSE error; convention is a `data: {error: ...}`
    line followed by a `data: [DONE]` terminator so the client unwinds cleanly."""
    payload = json.dumps({"error": {"type": "api_error", "message": message}})
    return f"data: {payload}\n\ndata: [DONE]\n\n".encode()


def _openai_responses_error_frame(message: str) -> bytes:
    """OpenAI Responses uses named SSE events; mirror with `event: error`."""
    payload = json.dumps({"type": "error", "message": message})
    return f"event: error\ndata: {payload}\n\n".encode()


_FRAMERS = {
    "anthropic": _anthropic_error_frame,
    "openai-chat": _openai_chat_error_frame,
    "openai-responses": _openai_responses_error_frame,
}


async def safe_stream(
    inner: AsyncIterator[bytes],
    *,
    shape: Shape,
    endpoint: str,
    model: str,
) -> AsyncIterator[bytes]:
    """Forward `inner` chunks; on exception, log + emit a per-shape SSE error."""
    try:
        async for chunk in inner:
            yield chunk
    except Exception as exc:
        log.warning(
            "egress.stream_error",
            shape=shape.name,
            endpoint=endpoint,
            model=model,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        framer = _FRAMERS.get(shape.name)
        if framer is not None:
            yield framer(str(exc))
