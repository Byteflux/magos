"""Soft-delete state machine. Pure: prior + fresh + clock -> next entries.

Missing-from-fresh marks ``deprecated_at``; reappearance clears it; past
``grace_seconds`` hard-deletes. See ``docs/registry/overview.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timedelta

from magos.registry.state import ModelEntry


def apply_deprecation(
    *,
    provider: str,
    prev_entries: Mapping[str, ModelEntry],
    fresh_entries: Mapping[str, ModelEntry],
    now: datetime,
    grace_seconds: int,
) -> dict[str, ModelEntry]:
    """Return next entries for ``provider`` after merging fresh against prev.

    ``prev_entries`` and ``fresh_entries`` are both keyed by namespaced
    id and may contain entries from any provider; this function only
    operates on entries whose ``provider`` field matches ``provider``.
    Entries from other providers are passed through unchanged. This lets
    callers feed in the whole registry and replace one provider's slice
    in place.
    """
    grace = timedelta(seconds=grace_seconds)
    next_entries: dict[str, ModelEntry] = {}

    # Pass-through: entries from other providers are untouched.
    for key, entry in prev_entries.items():
        if entry.provider != provider:
            next_entries[key] = entry

    # New + still-present entries from fresh: clear any stale deprecation mark.
    for key, fresh in fresh_entries.items():
        if fresh.provider != provider:
            continue
        if fresh.deprecated_at is None:
            next_entries[key] = fresh
        else:
            next_entries[key] = replace(fresh, deprecated_at=None)

    # Entries previously seen for this provider that fresh didn't mention.
    for key, prev in prev_entries.items():
        if prev.provider != provider or key in fresh_entries:
            continue
        if prev.deprecated_at is None:
            next_entries[key] = replace(prev, deprecated_at=now)
            continue
        if now - prev.deprecated_at >= grace:
            # Past the grace window: hard-delete by omission.
            continue
        next_entries[key] = prev

    return next_entries
