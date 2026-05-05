"""Coercion helpers shared across discovery adapters.

All functions treat ``bool`` as non-numeric to avoid Python's subclass
relationship between ``bool`` and ``int``.
"""

from __future__ import annotations

from typing import Any


def coerce_int(value: Any) -> int | None:
    """Return ``value`` coerced to ``int``, or ``None`` if not representable."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def coerce_float(value: Any) -> float | None:
    """Return ``value`` coerced to ``float``, or ``None`` if not representable."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def per_token_to_per_million(value: float | None) -> float | None:
    """Scale a per-token USD cost to per-million USD, dropping negative sentinels.

    Negative values signal "varies by underlying model" (OpenRouter meta
    routes) or similar; treat as unknown.
    """
    if value is None or value < 0:
        return None
    return value * 1_000_000
