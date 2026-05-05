"""Core registry data shapes (``ModelEntry`` + immutable ``RegistryState``).

See ``docs/registry/overview.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class ModelEntry:
    """One model, fully resolved across override / discovery / litellm."""

    provider: str
    raw_id: str
    litellm_id: str
    context_size: int | None = None
    max_output: int | None = None
    # USD per million tokens; adapters scale per-token upstream values.
    input_cost: float | None = None
    output_cost: float | None = None
    # Anthropic prompt-cache rates (read/write). OpenAI exposes only
    # cache reads. Sources without cache pricing leave these ``None`` and
    # consumers should fall back to ``input_cost``.
    cache_read_cost: float | None = None
    cache_write_cost: float | None = None
    input_modalities: tuple[str, ...] = ()
    output_modalities: tuple[str, ...] = ()
    deprecated_at: datetime | None = None
    sources: tuple[str, ...] = ()

    @property
    def namespaced_id(self) -> str:
        """Registry key: ``<provider>/<raw_id>``."""
        return f"{self.provider}/{self.raw_id}"

    @property
    def is_deprecated(self) -> bool:
        return self.deprecated_at is not None


def _freeze_entries(entries: Mapping[str, ModelEntry]) -> Mapping[str, ModelEntry]:
    return MappingProxyType(dict(entries))


def _freeze_by_provider(
    entries: Mapping[str, ModelEntry],
) -> Mapping[str, frozenset[str]]:
    by_provider: dict[str, set[str]] = {}
    for key, entry in entries.items():
        by_provider.setdefault(entry.provider, set()).add(key)
    return MappingProxyType({k: frozenset(v) for k, v in by_provider.items()})


def _freeze_refreshed(refreshed: Mapping[str, datetime]) -> Mapping[str, datetime]:
    return MappingProxyType(dict(refreshed))


@dataclass(frozen=True, slots=True)
class RegistryState:
    """Immutable snapshot of the registry; replaced atomically on refresh."""

    entries: Mapping[str, ModelEntry] = field(default_factory=lambda: MappingProxyType({}))
    refreshed_at: Mapping[str, datetime] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        # Defensive freezing: callers may pass plain dicts; we wrap so the
        # resulting state can't be mutated through aliases held by callers.
        object.__setattr__(self, "entries", _freeze_entries(self.entries))
        object.__setattr__(self, "refreshed_at", _freeze_refreshed(self.refreshed_at))

    @property
    def by_provider(self) -> Mapping[str, frozenset[str]]:
        """Index of namespaced ids grouped by provider, computed on demand."""
        return _freeze_by_provider(self.entries)

    def get(self, namespaced_id: str) -> ModelEntry | None:
        return self.entries.get(namespaced_id)

    def for_provider(self, provider: str) -> tuple[ModelEntry, ...]:
        keys = self.by_provider.get(provider, frozenset())
        return tuple(self.entries[k] for k in sorted(keys))
