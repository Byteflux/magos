"""Tests for ``magos.registry.store`` JSON persistence."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from magos.registry.state import ModelEntry, RegistryState
from magos.registry.store import deserialize, load, save, serialize


def _entry(provider: str = "openrouter", raw_id: str = "anthropic/claude-sonnet-4-6") -> ModelEntry:
    return ModelEntry(
        provider=provider,
        raw_id=raw_id,
        litellm_id=f"{provider}/{raw_id}",
        context_size=200000,
        max_output=8192,
        input_cost=3.0,
        output_cost=15.0,
        modalities=("text", "image"),
        sources=("discovery", "litellm"),
    )


def _state_with(*entries: ModelEntry) -> RegistryState:
    return RegistryState(
        entries={e.namespaced_id: e for e in entries},
        refreshed_at={e.provider: datetime(2026, 5, 2, 12, 0, tzinfo=UTC) for e in entries},
    )


def test_serialize_deserialize_round_trip_preserves_state() -> None:
    state = _state_with(_entry(), _entry("anthropic", "claude-sonnet-4-6"))
    raw = serialize(state)
    rebuilt = deserialize(raw)
    assert set(rebuilt.entries) == set(state.entries)
    for key in state.entries:
        assert rebuilt.entries[key] == state.entries[key]
    assert rebuilt.refreshed_at == state.refreshed_at


def test_serialize_handles_deprecated_entries() -> None:
    deprecated = ModelEntry(
        provider="openrouter",
        raw_id="legacy",
        litellm_id="openrouter/legacy",
        deprecated_at=datetime(2026, 4, 30, 0, 0, tzinfo=UTC),
    )
    state = _state_with(deprecated)
    rebuilt = deserialize(serialize(state))
    assert rebuilt.entries["openrouter/legacy"].deprecated_at == deprecated.deprecated_at


def test_load_returns_empty_state_when_file_missing(tmp_path: Path) -> None:
    state = load(tmp_path / "absent.json")
    assert state.entries == {}
    assert state.refreshed_at == {}


def test_load_returns_empty_state_on_corrupt_file(tmp_path: Path) -> None:
    target = tmp_path / "models.json"
    target.write_bytes(b"{not valid json")
    state = load(target)
    assert state.entries == {}


def test_save_then_load_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "models.json"
    state = _state_with(_entry())
    save(state, target)
    rebuilt = load(target)
    assert set(rebuilt.entries) == set(state.entries)


def test_save_creates_parent_directory(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "dir" / "models.json"
    save(_state_with(_entry()), target)
    assert target.exists()


def test_save_replaces_existing_file_atomically(tmp_path: Path) -> None:
    """A second save should replace the first; no temp file should linger."""
    target = tmp_path / "models.json"
    save(_state_with(_entry("openrouter", "a")), target)
    save(_state_with(_entry("openrouter", "b")), target)
    rebuilt = load(target)
    assert set(rebuilt.entries) == {"openrouter/b"}
    siblings = list(target.parent.iterdir())
    assert all(p.suffix != ".tmp" for p in siblings)


def test_load_returns_empty_when_required_field_missing(tmp_path: Path) -> None:
    target = tmp_path / "models.json"
    target.write_bytes(b'{"entries":[{"provider":"x"}]}')  # missing raw_id, litellm_id
    state = load(target)
    assert state.entries == {}
