"""Per-shape usage extractors + the canonical ``egress.usage`` log event.

The ``Usage`` dataclass canonicalises Anthropic / OpenAI Chat / OpenAI
Responses token counts into one shape; the per-shape ``usage_from_*``
extractors do the wire-format-specific parsing. ``log_usage_from_body``
is the convenience wrapper used by the non-streaming response path.
``cache_write`` is Anthropic-only; OpenAI shapes leave it 0.
"""

from __future__ import annotations

from collections.abc import Callable
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
