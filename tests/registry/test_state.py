"""Tests for `magos.registry.state` core data shapes."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from magos.registry.state import ModelEntry, RegistryState


def _entry(provider: str = "openrouter", raw_id: str = "anthropic/claude-sonnet-4-6") -> ModelEntry:
    return ModelEntry(
        provider=provider,
        raw_id=raw_id,
        litellm_id=f"{provider}/{raw_id}",
        context_size=200000,
    )


def test_model_entry_namespaced_id_combines_provider_and_raw_id() -> None:
    entry = _entry()
    assert entry.namespaced_id == "openrouter/anthropic/claude-sonnet-4-6"


def test_model_entry_is_deprecated_reflects_timestamp() -> None:
    entry = _entry()
    assert entry.is_deprecated is False
    deprecated = ModelEntry(
        provider="x",
        raw_id="y",
        litellm_id="x/y",
        deprecated_at=datetime.now(UTC),
    )
    assert deprecated.is_deprecated is True


def test_model_entry_is_frozen() -> None:
    entry = _entry()
    with pytest.raises(FrozenInstanceError):
        entry.context_size = 1000  # type: ignore[misc]


def test_registry_state_defaults_to_empty() -> None:
    state = RegistryState()
    assert state.entries == {}
    assert state.refreshed_at == {}
    assert state.by_provider == {}


def test_registry_state_get_returns_none_for_missing() -> None:
    state = RegistryState()
    assert state.get("openrouter/missing") is None


def test_registry_state_get_returns_entry() -> None:
    entry = _entry()
    state = RegistryState(entries={entry.namespaced_id: entry})
    assert state.get(entry.namespaced_id) is entry


def test_registry_state_for_provider_groups_entries() -> None:
    a = _entry("openrouter", "anthropic/claude-sonnet-4-6")
    b = _entry("openrouter", "openai/gpt-4o")
    c = _entry("anthropic", "claude-sonnet-4-6")
    state = RegistryState(entries={e.namespaced_id: e for e in (a, b, c)})
    assert {e.namespaced_id for e in state.for_provider("openrouter")} == {
        a.namespaced_id,
        b.namespaced_id,
    }
    assert state.for_provider("anthropic") == (c,)
    assert state.for_provider("missing") == ()


def test_registry_state_freezes_input_dicts() -> None:
    """Caller-held dicts must not mutate the snapshot."""
    entry = _entry()
    src = {entry.namespaced_id: entry}
    state = RegistryState(entries=src)
    src["other"] = entry  # mutate caller's dict
    assert "other" not in state.entries


def test_registry_state_by_provider_is_idempotent() -> None:
    entry = _entry()
    state = RegistryState(entries={entry.namespaced_id: entry})
    first = state.by_provider
    second = state.by_provider
    assert first == second


# ---------------------------------------------------------------------------
# resolve_for_dispatch
# ---------------------------------------------------------------------------


def _state_with(*entries: ModelEntry) -> RegistryState:
    return RegistryState(entries={e.namespaced_id: e for e in entries})


def test_resolve_for_dispatch_exact_namespaced_match() -> None:
    """Namespaced id resolves to the entry's litellm_id."""
    entry = _entry("openrouter", "anthropic/claude-sonnet-4-6")
    state = _state_with(entry)
    result = state.resolve_for_dispatch(entry.namespaced_id, provider=None)
    assert result == entry.litellm_id


def test_resolve_for_dispatch_raw_id_within_provider() -> None:
    """Bare raw_id with a matching provider resolves via <provider>/<raw_id>."""
    entry = _entry("openrouter", "anthropic/claude-sonnet-4-6")
    state = _state_with(entry)
    # model sent in request is just the raw_id; provider comes from the rule action
    result = state.resolve_for_dispatch(entry.raw_id, provider="openrouter")
    assert result == entry.litellm_id


def test_resolve_for_dispatch_miss_returns_none() -> None:
    """Unknown model with no matching entry returns None."""
    state = _state_with(_entry("openrouter", "anthropic/claude-sonnet-4-6"))
    assert state.resolve_for_dispatch("unknown/model", provider="openrouter") is None


def test_resolve_for_dispatch_provider_none_skips_raw_scan() -> None:
    """When provider is None, raw-id scan is skipped; returns None for bare raw_id."""
    entry = _entry("openrouter", "anthropic/claude-sonnet-4-6")
    state = _state_with(entry)
    # Without a provider the raw_id alone cannot be resolved
    assert state.resolve_for_dispatch(entry.raw_id, provider=None) is None


def test_resolve_for_dispatch_namespaced_beats_raw_for_same_string() -> None:
    """Exact namespaced match wins even when provider is also supplied."""
    entry = _entry("openrouter", "anthropic/claude-sonnet-4-6")
    state = _state_with(entry)
    result = state.resolve_for_dispatch(entry.namespaced_id, provider="openrouter")
    assert result == entry.litellm_id
