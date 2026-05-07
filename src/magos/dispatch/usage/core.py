"""``log_usage`` + ``log_usage_from_body``: canonical ``egress.usage`` log event.

The ``Usage`` dataclass and the per-shape extraction logic live in
:mod:`magos.shapes` (``Shape.extract_usage`` is the canonical extractor;
``Usage`` is re-exported here for backward-compatible imports inside
:mod:`magos.dispatch.usage`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from magos.shapes import Shape, Usage
from magos.telemetry import get_logger

log = get_logger("magos.dispatch.usage")


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
        shape=shape.name,
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
    usage = shape.extract_usage(body)
    log_usage(shape, endpoint=endpoint, model=model, usage=usage, stream=stream)
    if on_complete is not None and not usage.is_empty:
        on_complete(usage)
    return usage


__all__ = ["Usage", "log_usage", "log_usage_from_body"]
