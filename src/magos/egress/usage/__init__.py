"""Per-response token-usage logging across Anthropic / OpenAI shapes.

Three concerns split across siblings:

- :mod:`core` — the ``Usage`` dataclass + per-shape extractors +
  ``log_usage`` / ``log_usage_from_body`` (the non-streaming path).
- :mod:`accumulator` — ``UsageAccumulator``, the per-shape SSE event
  aggregator used during streaming.
- :mod:`tap` — ``tap_stream``, the byte-passthrough generator that
  feeds the accumulator and emits the final ``egress.usage`` log.

``cache_write`` is Anthropic-only; OpenAI shapes leave it 0. The
``Shape`` literal and ``shape_for_endpoint`` lookup live in
:mod:`magos.shapes`.
"""

from __future__ import annotations

from .accumulator import UsageAccumulator
from .core import (
    Usage,
    log_usage,
    log_usage_from_body,
    usage_from_anthropic,
    usage_from_openai_chat,
    usage_from_openai_responses,
)
from .tap import tap_stream

__all__ = [
    "Usage",
    "UsageAccumulator",
    "log_usage",
    "log_usage_from_body",
    "tap_stream",
    "usage_from_anthropic",
    "usage_from_openai_chat",
    "usage_from_openai_responses",
]
