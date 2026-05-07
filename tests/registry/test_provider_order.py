"""Tests for `magos.registry.provider_order` tie-break."""

from __future__ import annotations

from magos.registry.provider_order import resolve_provider


def test_returns_none_when_no_candidates() -> None:
    assert resolve_provider(raw_id="x", candidates=[]) is None


def test_pin_wins_when_in_candidates() -> None:
    out = resolve_provider(
        raw_id="claude-sonnet-4-6",
        candidates={"openrouter", "anthropic"},
        pins={"claude-sonnet-4-6": "anthropic"},
        provider_order=("openrouter", "anthropic"),
    )
    assert out == "anthropic"


def test_pin_ignored_when_not_in_candidates() -> None:
    out = resolve_provider(
        raw_id="claude-sonnet-4-6",
        candidates={"openrouter"},
        pins={"claude-sonnet-4-6": "bedrock"},
        provider_order=("openrouter", "anthropic"),
    )
    assert out == "openrouter"


def test_provider_order_wins_over_first_registered() -> None:
    out = resolve_provider(
        raw_id="x",
        candidates={"zeta", "alpha", "openrouter"},
        provider_order=("openrouter", "alpha"),
    )
    assert out == "openrouter"


def test_first_registered_falls_back_to_lex_smallest() -> None:
    out = resolve_provider(
        raw_id="x",
        candidates={"zeta", "alpha", "middle"},
        provider_order=(),
    )
    assert out == "alpha"


def test_provider_order_partial_overlap_picks_first_match() -> None:
    out = resolve_provider(
        raw_id="x",
        candidates={"alpha", "beta"},
        provider_order=("missing", "beta", "alpha"),
    )
    assert out == "beta"
