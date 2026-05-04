"""Tests for ``magos.registry.deprecation`` state machine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from magos.registry.deprecation import apply_deprecation
from magos.registry.state import ModelEntry


def _entry(provider: str, raw_id: str, deprecated_at: datetime | None = None) -> ModelEntry:
    return ModelEntry(
        provider=provider,
        raw_id=raw_id,
        litellm_id=f"{provider}/{raw_id}",
        deprecated_at=deprecated_at,
    )


def test_new_model_appears_in_fresh_set() -> None:
    now = datetime(2026, 5, 2, tzinfo=UTC)
    fresh = {"p/a": _entry("p", "a")}
    out = apply_deprecation(
        provider="p",
        prev_entries={},
        fresh_entries=fresh,
        now=now,
        grace_seconds=86400,
    )
    assert "p/a" in out
    assert out["p/a"].deprecated_at is None


def test_missing_model_gets_marked_deprecated_now() -> None:
    now = datetime(2026, 5, 2, tzinfo=UTC)
    prev = {"p/a": _entry("p", "a")}
    out = apply_deprecation(
        provider="p",
        prev_entries=prev,
        fresh_entries={},
        now=now,
        grace_seconds=86400,
    )
    assert out["p/a"].deprecated_at == now


def test_reappeared_model_clears_deprecation() -> None:
    earlier = datetime(2026, 5, 1, tzinfo=UTC)
    now = datetime(2026, 5, 2, tzinfo=UTC)
    prev = {"p/a": _entry("p", "a", deprecated_at=earlier)}
    fresh = {"p/a": _entry("p", "a")}
    out = apply_deprecation(
        provider="p",
        prev_entries=prev,
        fresh_entries=fresh,
        now=now,
        grace_seconds=86400,
    )
    assert out["p/a"].deprecated_at is None


def test_past_grace_window_is_hard_deleted() -> None:
    deprecated_at = datetime(2026, 5, 1, tzinfo=UTC)
    now = deprecated_at + timedelta(days=4)  # grace is 3 days
    prev = {"p/a": _entry("p", "a", deprecated_at=deprecated_at)}
    out = apply_deprecation(
        provider="p",
        prev_entries=prev,
        fresh_entries={},
        now=now,
        grace_seconds=3 * 86400,
    )
    assert "p/a" not in out


def test_within_grace_window_preserved_with_existing_mark() -> None:
    deprecated_at = datetime(2026, 5, 1, tzinfo=UTC)
    now = deprecated_at + timedelta(days=2)  # grace is 3 days
    prev = {"p/a": _entry("p", "a", deprecated_at=deprecated_at)}
    out = apply_deprecation(
        provider="p",
        prev_entries=prev,
        fresh_entries={},
        now=now,
        grace_seconds=3 * 86400,
    )
    assert out["p/a"].deprecated_at == deprecated_at


def test_other_provider_entries_pass_through_untouched() -> None:
    now = datetime(2026, 5, 2, tzinfo=UTC)
    prev = {
        "openrouter/x": _entry("openrouter", "x"),
        "anthropic/y": _entry("anthropic", "y"),
    }
    out = apply_deprecation(
        provider="openrouter",
        prev_entries=prev,
        fresh_entries={},
        now=now,
        grace_seconds=86400,
    )
    assert out["anthropic/y"] is prev["anthropic/y"]
    assert out["openrouter/x"].deprecated_at == now


def test_grace_window_boundary_inclusive() -> None:
    deprecated_at = datetime(2026, 5, 1, tzinfo=UTC)
    grace = 3 * 86400
    now_at_boundary = deprecated_at + timedelta(seconds=grace)
    prev = {"p/a": _entry("p", "a", deprecated_at=deprecated_at)}
    out = apply_deprecation(
        provider="p",
        prev_entries=prev,
        fresh_entries={},
        now=now_at_boundary,
        grace_seconds=grace,
    )
    # At exactly the grace duration, the entry is hard-deleted.
    assert "p/a" not in out
