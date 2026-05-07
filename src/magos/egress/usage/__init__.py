"""Per-response token-usage logging across Anthropic / OpenAI shapes.

Three concerns split across siblings:

- :mod:`core` — ``log_usage`` / ``log_usage_from_body`` (the latter is
  a thin convenience over ``Shape.extract_usage`` plus logging).
- :mod:`accumulator` — ``UsageAccumulator``, the per-shape SSE event
  aggregator used during streaming.
- :mod:`tap` — ``tap_stream``, the byte-passthrough generator that
  feeds the accumulator and emits the final ``egress.usage`` log.

The ``Usage`` dataclass lives in :mod:`magos.shapes.usage` and is
re-exported here for backward-compatible local imports.
"""

from __future__ import annotations

from magos.shapes import Usage

from .accumulator import UsageAccumulator
from .core import log_usage, log_usage_from_body
from .tap import tap_stream

__all__ = [
    "Usage",
    "UsageAccumulator",
    "log_usage",
    "log_usage_from_body",
    "tap_stream",
]
