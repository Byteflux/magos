"""Tie-breaking when multiple providers serve the same logical model.

A request like ``{"model": "openrouter/anthropic/claude-sonnet-4-6"}`` is
already namespaced and never needs tie-breaking. But auto-routing matches
by ``raw_id`` across providers, and a logical model id like
``claude-sonnet-4-6`` may resolve to several registry entries.

Resolution chain (highest priority first):

    1. ``pins[raw_id]``       - explicit per-model pin in magos.yaml
    2. ``provider_order``     - global preference order; first-listed wins
    3. first-registered       - deterministic fallback by sorted provider name

If none of the candidate providers appear in ``provider_order`` and no pin
matches, ``resolve_provider`` returns the candidate with the lexicographically
smallest provider name. Stable order matters for routing reproducibility.
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
    """Pick the winning provider from ``candidates``, or ``None`` if empty.

    ``pins`` maps ``raw_id`` to a pinned provider name; if the pin matches
    a candidate, it wins outright. Pins that point to a provider not in
    candidates are ignored (the pinned provider doesn't serve this model).
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
