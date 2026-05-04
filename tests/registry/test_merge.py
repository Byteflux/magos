"""Tests for ``magos.registry.merge`` precedence behavior."""

from __future__ import annotations

from magos.registry.litellm_lookup import PartialEntry
from magos.registry.merge import merge


def test_merge_uses_default_litellm_id_when_no_source_supplies_one() -> None:
    entry = merge(
        provider="openrouter",
        raw_id="anthropic/claude-sonnet-4-6",
        default_litellm_id="openrouter/anthropic/claude-sonnet-4-6",
    )
    assert entry.litellm_id == "openrouter/anthropic/claude-sonnet-4-6"
    assert entry.sources == ()
    assert entry.context_size is None


def test_merge_override_wins_over_discovery_and_litellm() -> None:
    entry = merge(
        provider="openrouter",
        raw_id="anthropic/claude-sonnet-4-6",
        default_litellm_id="openrouter/anthropic/claude-sonnet-4-6",
        override=PartialEntry(context_size=1_000_000),
        discovered=PartialEntry(context_size=200_000),
        litellm_fallback=PartialEntry(context_size=128_000),
    )
    assert entry.context_size == 1_000_000
    assert entry.sources == ("override", "discovery", "litellm")


def test_merge_discovery_wins_over_litellm_when_no_override() -> None:
    entry = merge(
        provider="openrouter",
        raw_id="anthropic/claude-sonnet-4-6",
        default_litellm_id="openrouter/anthropic/claude-sonnet-4-6",
        discovered=PartialEntry(context_size=200_000),
        litellm_fallback=PartialEntry(context_size=128_000),
    )
    assert entry.context_size == 200_000
    assert entry.sources == ("discovery", "litellm")


def test_merge_litellm_fills_gap_when_discovery_missing_field() -> None:
    entry = merge(
        provider="openrouter",
        raw_id="anthropic/claude-sonnet-4-6",
        default_litellm_id="openrouter/anthropic/claude-sonnet-4-6",
        discovered=PartialEntry(context_size=200_000),  # no costs
        litellm_fallback=PartialEntry(input_cost=3.0, output_cost=15.0),
    )
    assert entry.context_size == 200_000
    assert entry.input_cost == 3.0
    assert entry.output_cost == 15.0


def test_merge_override_can_replace_litellm_id() -> None:
    entry = merge(
        provider="openrouter",
        raw_id="anthropic/claude-sonnet-4-6",
        default_litellm_id="openrouter/anthropic/claude-sonnet-4-6",
        override=PartialEntry(litellm_id="openrouter/anthropic/claude-sonnet-4-6:1m"),
    )
    assert entry.litellm_id == "openrouter/anthropic/claude-sonnet-4-6:1m"


def test_merge_threads_cache_costs_with_same_precedence_as_input_output() -> None:
    """Cache costs follow override > discovery > litellm, like input/output_cost."""
    entry = merge(
        provider="openrouter",
        raw_id="anthropic/claude-sonnet-4-6",
        default_litellm_id="openrouter/anthropic/claude-sonnet-4-6",
        # Discovery contributes cache_read_cost, litellm fills cache_write_cost.
        discovered=PartialEntry(input_cost=3.0, cache_read_cost=0.30),
        litellm_fallback=PartialEntry(input_cost=2.99, cache_read_cost=0.40, cache_write_cost=3.75),
    )
    assert entry.cache_read_cost == 0.30
    assert entry.cache_write_cost == 3.75


def test_merge_modalities_take_first_non_none_tuple() -> None:
    entry = merge(
        provider="openrouter",
        raw_id="anthropic/claude-sonnet-4-6",
        default_litellm_id="x",
        discovered=PartialEntry(input_modalities=("text", "image"), output_modalities=("text",)),
        litellm_fallback=PartialEntry(input_modalities=("text",), output_modalities=("text",)),
    )
    assert entry.input_modalities == ("text", "image")
    assert entry.output_modalities == ("text",)


def test_merge_only_lists_sources_that_actually_contributed() -> None:
    """An empty PartialEntry on a source shouldn't appear in sources."""
    entry = merge(
        provider="openrouter",
        raw_id="x",
        default_litellm_id="openrouter/x",
        override=PartialEntry(),  # nothing set
        discovered=PartialEntry(context_size=100),
    )
    assert entry.sources == ("discovery",)


def test_merge_handles_zero_cost_correctly() -> None:
    """0.0 is a meaningful value, not "missing" — must not be skipped."""
    entry = merge(
        provider="local",
        raw_id="vllm-llama",
        default_litellm_id="local/vllm-llama",
        discovered=PartialEntry(input_cost=0.0, output_cost=0.0),
        litellm_fallback=PartialEntry(input_cost=1.0, output_cost=1.0),
    )
    assert entry.input_cost == 0.0
    assert entry.output_cost == 0.0
