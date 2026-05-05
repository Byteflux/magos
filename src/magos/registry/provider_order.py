"""Tie-break when multiple providers serve one logical model id.

Resolution: pin > ``provider_order`` > lex-smallest candidate. See
``docs/registry/auto-routing.md``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping


def resolve_provider(
    *,
    raw_id: str,
    candidates: Iterable[str],
    pins: Mapping[str, str] | None = None,
    provider_order: tuple[str, ...] = (),
) -> str | None:
    """Pick the winning provider from ``candidates`` (or ``None`` if empty).

    Pins to providers absent from ``candidates`` are ignored.
    """
    candidate_set = set(candidates)
    if not candidate_set:
        return None

    if pins is not None:
        pinned = pins.get(raw_id)
        if pinned and pinned in candidate_set:
            return pinned

    for provider in provider_order:
        if provider in candidate_set:
            return provider

    return min(candidate_set)
