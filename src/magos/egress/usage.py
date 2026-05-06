"""Per-response token-usage logging across Anthropic / OpenAI shapes.

Streaming uses a ``UsageAccumulator`` fed by an in-line SSE parser; the
byte stream is forwarded verbatim regardless. ``cache_write`` is
Anthropic-only; OpenAI shapes leave it 0.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, Literal

from magos.telemetry import get_logger

log = get_logger("magos.egress.usage")

Shape = Literal["anthropic", "openai-chat", "openai-responses"]


@dataclass(frozen=True, slots=True)
class Usage:
    """Canonicalised token counts for one request."""

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0

    @property
    def is_empty(self) -> bool:
        return (
            self.input == 0 and self.output == 0 and self.cache_read == 0 and self.cache_write == 0
        )


def _safe_int(value: Any) -> int:
    return value if isinstance(value, int) and value >= 0 else 0


def usage_from_anthropic(body: Any) -> Usage:
    """Extract usage from an Anthropic Messages response dict."""
    if not isinstance(body, dict):
        return Usage()
    u = body.get("usage")
    if not isinstance(u, dict):
        return Usage()
    return Usage(
        input=_safe_int(u.get("input_tokens")),
        output=_safe_int(u.get("output_tokens")),
        cache_read=_safe_int(u.get("cache_read_input_tokens")),
        cache_write=_safe_int(u.get("cache_creation_input_tokens")),
    )


def usage_from_openai_chat(body: Any) -> Usage:
    """Extract usage from an OpenAI Chat Completions response dict."""
    if not isinstance(body, dict):
        return Usage()
    u = body.get("usage")
    if not isinstance(u, dict):
        return Usage()
    details = u.get("prompt_tokens_details")
    cache_read = _safe_int(details.get("cached_tokens")) if isinstance(details, dict) else 0
    return Usage(
        input=_safe_int(u.get("prompt_tokens")),
        output=_safe_int(u.get("completion_tokens")),
        cache_read=cache_read,
    )


def usage_from_openai_responses(body: Any) -> Usage:
    """Extract usage from an OpenAI Responses response dict."""
    if not isinstance(body, dict):
        return Usage()
    u = body.get("usage")
    if not isinstance(u, dict):
        return Usage()
    details = u.get("input_tokens_details")
    cache_read = _safe_int(details.get("cached_tokens")) if isinstance(details, dict) else 0
    return Usage(
        input=_safe_int(u.get("input_tokens")),
        output=_safe_int(u.get("output_tokens")),
        cache_read=cache_read,
    )


_EXTRACTORS: dict[Shape, Callable[[Any], Usage]] = {
    "anthropic": usage_from_anthropic,
    "openai-chat": usage_from_openai_chat,
    "openai-responses": usage_from_openai_responses,
}


def shape_for_endpoint(endpoint: str) -> Shape | None:
    """Map a routed endpoint to the response shape, or ``None`` for n/a."""
    if endpoint == "/v1/messages":
        return "anthropic"
    if endpoint == "/v1/chat/completions":
        return "openai-chat"
    if endpoint in {"/v1/responses", "/v1/responses/{id}"}:
        return "openai-responses"
    return None


def log_usage(
    shape: Shape,
    *,
    endpoint: str,
    model: str | None,
    usage: Usage,
    stream: bool = False,
) -> None:
    """Emit ``egress.usage`` if any field is non-zero; no-op on empty usage."""
    if usage.is_empty:
        return
    log.info(
        "egress.usage",
        shape=shape,
        endpoint=endpoint,
        model=model,
        stream=stream,
        input=usage.input,
        output=usage.output,
        cache_read=usage.cache_read,
        cache_write=usage.cache_write,
    )


def log_usage_from_body(
    shape: Shape,
    body: Any,
    *,
    endpoint: str,
    stream: bool = False,
    on_complete: Callable[[Usage], None] | None = None,
) -> Usage:
    """Convenience: extract usage for ``shape`` from ``body``, log it, return it.

    If ``on_complete`` is provided and the captured usage is non-empty,
    it is invoked with the ``Usage``. The hook MUST NOT raise; callers
    that need failure isolation should wrap their callback themselves.
    """
    extractor = _EXTRACTORS[shape]
    model = body.get("model") if isinstance(body, dict) else None
    usage = extractor(body)
    log_usage(shape, endpoint=endpoint, model=model, usage=usage, stream=stream)
    if on_complete is not None and not usage.is_empty:
        on_complete(usage)
    return usage


# ---------------------------------------------------------------------------
# Streaming: SSE-tap accumulators
# ---------------------------------------------------------------------------


class UsageAccumulator:
    """Stateful usage accumulator fed parsed SSE events as the stream passes."""

    def __init__(self, shape: Shape) -> None:
        self._shape = shape
        self._input = 0
        self._output = 0
        self._cache_read = 0
        self._cache_write = 0
        self._model: str | None = None

    @property
    def model(self) -> str | None:
        return self._model

    def snapshot(self) -> Usage:
        return Usage(
            input=self._input,
            output=self._output,
            cache_read=self._cache_read,
            cache_write=self._cache_write,
        )

    def feed(self, event_name: str | None, data: dict[str, Any]) -> None:
        if self._shape == "anthropic":
            self._feed_anthropic(event_name, data)
        elif self._shape == "openai-chat":
            self._feed_openai_chat(data)
        else:
            self._feed_openai_responses(event_name, data)

    def _feed_anthropic(self, event_name: str | None, data: dict[str, Any]) -> None:
        # Input + cache arrive on ``message_start.message.usage``; final
        # output arrives on ``message_delta.usage``.
        if event_name == "message_start":
            message = data.get("message")
            if isinstance(message, dict):
                u = message.get("usage")
                if isinstance(u, dict):
                    self._input = _safe_int(u.get("input_tokens"))
                    self._cache_read = _safe_int(u.get("cache_read_input_tokens"))
                    self._cache_write = _safe_int(u.get("cache_creation_input_tokens"))
                model = message.get("model")
                if isinstance(model, str):
                    self._model = model
        elif event_name == "message_delta":
            u = data.get("usage")
            if isinstance(u, dict):
                output = _safe_int(u.get("output_tokens"))
                if output:
                    self._output = output

    def _feed_openai_chat(self, data: dict[str, Any]) -> None:
        # Usage only on the terminal chunk, gated on
        # ``stream_options.include_usage: true``.
        u = data.get("usage")
        if isinstance(u, dict):
            self._input = _safe_int(u.get("prompt_tokens"))
            self._output = _safe_int(u.get("completion_tokens"))
            details = u.get("prompt_tokens_details")
            if isinstance(details, dict):
                self._cache_read = _safe_int(details.get("cached_tokens"))
        model = data.get("model")
        if isinstance(model, str):
            self._model = model

    def _feed_openai_responses(self, event_name: str | None, data: dict[str, Any]) -> None:
        # Usage arrives on ``response.completed.response.usage``.
        if event_name == "response.completed":
            response = data.get("response")
            if isinstance(response, dict):
                u = response.get("usage")
                if isinstance(u, dict):
                    self._input = _safe_int(u.get("input_tokens"))
                    self._output = _safe_int(u.get("output_tokens"))
                    details = u.get("input_tokens_details")
                    if isinstance(details, dict):
                        self._cache_read = _safe_int(details.get("cached_tokens"))
                model = response.get("model")
                if isinstance(model, str):
                    self._model = model


# ---------------------------------------------------------------------------
# SSE byte-stream parser
# ---------------------------------------------------------------------------


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
