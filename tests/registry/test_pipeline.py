"""Tests for ``magos.registry.pipeline`` pure functions."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from magos.registry.discovery.base import DiscoveredModel, DiscoveryResult
from magos.registry.litellm_lookup import PartialEntry
from magos.registry.pipeline import (
    ProviderDiff,
    diff_provider,
    merge_provider,
    override_to_partial,
)
from magos.registry.schema import ModelOverride, ProviderConfig
from magos.registry.state import ModelEntry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _provider_cfg(**kwargs: Any) -> ProviderConfig:
    return ProviderConfig.model_validate(kwargs)


def _no_litellm(model: str) -> dict[str, Any]:
    raise ValueError("not in litellm registry")


def _entry(provider: str, raw_id: str, **kwargs: Any) -> ModelEntry:
    return ModelEntry(
        provider=provider,
        raw_id=raw_id,
        litellm_id=kwargs.pop("litellm_id", f"{provider}/{raw_id}"),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# override_to_partial
# ---------------------------------------------------------------------------


def test_override_to_partial_none_returns_none() -> None:
    assert override_to_partial(None) is None


def test_override_to_partial_full_roundtrip() -> None:
    override = ModelOverride(
        context_size=128_000,
        max_output=4096,
        input_cost=3.0,
        output_cost=15.0,
        cache_read_cost=0.3,
        cache_write_cost=3.75,
        litellm_id="openrouter/anthropic/claude-sonnet-4-6",
        input_modalities=("text", "image"),
        output_modalities=("text",),
    )
    partial = override_to_partial(override)
    assert partial is not None
    assert partial.context_size == 128_000
    assert partial.max_output == 4096
    assert partial.input_cost == 3.0
    assert partial.output_cost == 15.0
    assert partial.cache_read_cost == 0.3
    assert partial.cache_write_cost == 3.75
    assert partial.litellm_id == "openrouter/anthropic/claude-sonnet-4-6"
    assert partial.input_modalities == ("text", "image")
    assert partial.output_modalities == ("text",)


def test_override_to_partial_partial_fields() -> None:
    override = ModelOverride(context_size=32_000)
    partial = override_to_partial(override)
    assert partial is not None
    assert partial.context_size == 32_000
    assert partial.litellm_id is None
    assert partial.max_output is None


# ---------------------------------------------------------------------------
# merge_provider -- with discovery
# ---------------------------------------------------------------------------


def test_merge_provider_discovery_hit() -> None:
    cfg = _provider_cfg(discovery="openrouter")
    result = DiscoveryResult(
        models=(
            DiscoveredModel(
                raw_id="anthropic/claude-sonnet-4-6",
                litellm_id="openrouter/anthropic/claude-sonnet-4-6",
                partial=PartialEntry(context_size=200_000),
            ),
        )
    )
    entries = merge_provider("openrouter", cfg, result, _no_litellm)
    assert "openrouter/anthropic/claude-sonnet-4-6" in entries
    entry = entries["openrouter/anthropic/claude-sonnet-4-6"]
    assert entry.context_size == 200_000
    assert entry.litellm_id == "openrouter/anthropic/claude-sonnet-4-6"


def test_merge_provider_override_takes_precedence_over_discovery() -> None:
    cfg = _provider_cfg(
        discovery="openrouter",
        models={
            "anthropic/claude-sonnet-4-6": ModelOverride(context_size=1_000_000).model_dump(),
        },
    )
    result = DiscoveryResult(
        models=(
            DiscoveredModel(
                raw_id="anthropic/claude-sonnet-4-6",
                litellm_id="openrouter/anthropic/claude-sonnet-4-6",
                partial=PartialEntry(context_size=200_000),
            ),
        )
    )
    entries = merge_provider("openrouter", cfg, result, _no_litellm)
    entry = entries["openrouter/anthropic/claude-sonnet-4-6"]
    assert entry.context_size == 1_000_000


def test_merge_provider_with_litellm_fallback() -> None:
    """LiteLLM lookup fills in fields not present in discovery."""

    def _mock_litellm(model: str) -> dict[str, Any]:
        if model == "openrouter/anthropic/claude-sonnet-4-6":
            return {"max_input_tokens": 200_000, "max_output_tokens": 8192}
        raise ValueError("unknown")

    cfg = _provider_cfg(discovery="openrouter")
    result = DiscoveryResult(
        models=(
            DiscoveredModel(
                raw_id="anthropic/claude-sonnet-4-6",
                litellm_id="openrouter/anthropic/claude-sonnet-4-6",
                partial=PartialEntry(),
            ),
        )
    )
    entries = merge_provider("openrouter", cfg, result, _mock_litellm)
    entry = entries["openrouter/anthropic/claude-sonnet-4-6"]
    assert entry.context_size == 200_000
    assert entry.max_output == 8192


# ---------------------------------------------------------------------------
# merge_provider -- empty discovery (manual entries only)
# ---------------------------------------------------------------------------


def test_merge_provider_manual_only_empty_discovery() -> None:
    cfg = _provider_cfg(
        litellm_provider="openai",
        models={
            "custom-model": ModelOverride(
                context_size=32_000, litellm_id="openai/custom"
            ).model_dump(),
        },
    )
    entries = merge_provider("manual", cfg, DiscoveryResult(), _no_litellm)
    assert "manual/custom-model" in entries
    entry = entries["manual/custom-model"]
    assert entry.context_size == 32_000
    assert entry.litellm_id == "openai/custom"


def test_merge_provider_manual_default_litellm_id_uses_litellm_provider() -> None:
    """Without an explicit litellm_id, default is ``<litellm_provider>/<raw_id>``."""
    cfg = _provider_cfg(
        litellm_provider="openai",
        models={
            "my-model": ModelOverride(context_size=8000).model_dump(),
        },
    )
    entries = merge_provider("myprovider", cfg, DiscoveryResult(), _no_litellm)
    entry = entries["myprovider/my-model"]
    assert entry.litellm_id == "openai/my-model"


def test_merge_provider_manual_default_litellm_id_falls_back_to_provider_name() -> None:
    """No litellm_provider set: default is ``<provider_name>/<raw_id>``."""
    cfg = _provider_cfg(
        models={
            "my-model": ModelOverride(context_size=8000).model_dump(),
        },
    )
    entries = merge_provider("myprovider", cfg, DiscoveryResult(), _no_litellm)
    entry = entries["myprovider/my-model"]
    assert entry.litellm_id == "myprovider/my-model"


# ---------------------------------------------------------------------------
# diff_provider
# ---------------------------------------------------------------------------


def test_diff_provider_all_added() -> None:
    prev: dict[str, ModelEntry] = {}
    nxt = {
        "p/a": _entry("p", "a"),
        "p/b": _entry("p", "b"),
    }
    diff = diff_provider("p", prev, nxt)
    assert isinstance(diff, ProviderDiff)
    assert diff.total == 2
    assert diff.added == 2
    assert diff.deprecated == 0
    assert diff.pruned == 0


def test_diff_provider_some_removed() -> None:
    prev = {
        "p/a": _entry("p", "a"),
        "p/b": _entry("p", "b"),
    }
    nxt = {"p/a": _entry("p", "a")}
    diff = diff_provider("p", prev, nxt)
    assert diff.total == 1
    assert diff.added == 0
    assert diff.pruned == 1


def test_diff_provider_deprecated_newly() -> None:
    dep_time = datetime(2026, 5, 1, tzinfo=UTC)
    prev = {"p/a": _entry("p", "a")}
    nxt = {"p/a": _entry("p", "a", deprecated_at=dep_time)}
    diff = diff_provider("p", prev, nxt)
    assert diff.deprecated == 1
    assert diff.added == 0
    assert diff.pruned == 0


def test_diff_provider_only_counts_own_provider() -> None:
    """Entries from a different provider must not affect the diff."""
    prev = {
        "p/a": _entry("p", "a"),
        "other/x": _entry("other", "x"),
    }
    nxt = {
        "p/a": _entry("p", "a"),
        "p/b": _entry("p", "b"),
        "other/x": _entry("other", "x"),
    }
    diff = diff_provider("p", prev, nxt)
    assert diff.total == 2
    assert diff.added == 1
    assert diff.pruned == 0


def test_diff_provider_no_changes() -> None:
    prev = {"p/a": _entry("p", "a")}
    diff = diff_provider("p", prev, prev)
    assert diff.total == 1
    assert diff.added == 0
    assert diff.deprecated == 0
    assert diff.pruned == 0
