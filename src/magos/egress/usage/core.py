"""Generic usage extractor + the canonical ``egress.usage`` log event.

``Usage`` canonicalises Anthropic / OpenAI Chat / OpenAI Responses token
counts into one shape; ``usage_from_body`` walks the per-shape
``usage_keys`` map from :mod:`magos.shapes` so the wire-format-specific
field names live there, not here. ``log_usage_from_body`` is the
convenience wrapper used by the non-streaming response path.
``cache_write`` is Anthropic-only; OpenAI shapes leave it 0.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from magos.shapes import SHAPES, Shape
from magos.telemetry import get_logger

log = get_logger("magos.egress.usage")


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


def _walk(body: Any, path: tuple[str, ...]) -> Any:
    """Walk a dotted path through nested dicts, returning ``None`` on any miss."""
    cur: Any = body
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def usage_from_body(shape: Shape, body: Any) -> Usage:
    """Extract usage from ``body`` using the per-shape ``usage_keys`` map.

    Reads :data:`magos.shapes.SHAPES[shape].usage_keys` to find the path
    to each canonical field; missing / non-int / negative values default
    to 0. Non-dict bodies return an empty ``Usage``.
    """
    if not isinstance(body, dict):
        return Usage()
    keys = SHAPES[shape].usage_keys
    return Usage(
        input=_safe_int(_walk(body, keys["input"])) if "input" in keys else 0,
        output=_safe_int(_walk(body, keys["output"])) if "output" in keys else 0,
        cache_read=_safe_int(_walk(body, keys["cache_read"])) if "cache_read" in keys else 0,
        cache_write=_safe_int(_walk(body, keys["cache_write"])) if "cache_write" in keys else 0,
    )


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
    model = body.get("model") if isinstance(body, dict) else None
    usage = usage_from_body(shape, body)
    log_usage(shape, endpoint=endpoint, model=model, usage=usage, stream=stream)
    if on_complete is not None and not usage.is_empty:
        on_complete(usage)
    return usage
