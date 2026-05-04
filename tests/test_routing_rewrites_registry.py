"""Tests that compress reads context_size from the registry."""

from __future__ import annotations

from magos.registry.state import ModelEntry, RegistryState
from magos.routing.rewrites import _resolve_model_limit


def _registry_with_context(model: str, size: int) -> RegistryState:
    entry = ModelEntry(
        provider=model.split("/", 1)[0],
        raw_id=model.split("/", 1)[1],
        litellm_id=model,
        context_size=size,
    )
    return RegistryState(entries={entry.namespaced_id: entry})


def test_registry_context_size_wins_over_litellm_default() -> None:
    registry = _registry_with_context("custom/llama-3-70b", 32768)
    assert _resolve_model_limit("custom/llama-3-70b", registry=registry) == 32768


def test_registry_miss_falls_back_to_litellm_or_default() -> None:
    registry = RegistryState()
    # LiteLLM may or may not know "fake/missing"; either way the fallback
    # path should not raise and should return some positive integer.
    result = _resolve_model_limit("fake/missing-model", registry=registry, default=200_000)
    assert isinstance(result, int)
    assert result >= 1024


def test_registry_entry_without_context_size_falls_through() -> None:
    entry = ModelEntry(
        provider="custom",
        raw_id="x",
        litellm_id="custom/x",
    )
    registry = RegistryState(entries={entry.namespaced_id: entry})
    result = _resolve_model_limit("custom/x", registry=registry, default=12345)
    # context_size missing on the entry, falls through to litellm/default.
    assert result == 12345 or result > 1024


def test_no_registry_uses_existing_litellm_path() -> None:
    """Backwards compat: omitting registry leaves prior behavior intact."""
    result = _resolve_model_limit("anthropic/claude-haiku-4-5", default=200_000)
    assert isinstance(result, int)
