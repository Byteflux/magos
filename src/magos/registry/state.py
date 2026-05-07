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


def _freeze_by_raw_id(
    entries: Mapping[str, ModelEntry],
) -> Mapping[str, frozenset[str]]:
    by_raw_id: dict[str, set[str]] = {}
    for entry in entries.values():
        by_raw_id.setdefault(entry.raw_id, set()).add(entry.provider)
    return MappingProxyType({k: frozenset(v) for k, v in by_raw_id.items()})


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

    @property
    def by_raw_id(self) -> Mapping[str, frozenset[str]]:
        """Index of providers serving each raw model id, computed on demand."""
        return _freeze_by_raw_id(self.entries)

    def get(self, namespaced_id: str) -> ModelEntry | None:
        return self.entries.get(namespaced_id)

    def for_provider(self, provider: str) -> tuple[ModelEntry, ...]:
        keys = self.by_provider.get(provider, frozenset())
        return tuple(self.entries[k] for k in sorted(keys))

    def providers_for_raw_id(self, raw_id: str) -> frozenset[str]:
        """Providers whose registry entries carry ``raw_id`` (empty if none)."""
        return self.by_raw_id.get(raw_id, frozenset())

    def find_by_model_id(self, model: str) -> ModelEntry | None:
        """Best-effort entry lookup for an inbound body model field.

        First tries an exact namespaced match (``<provider>/<raw_id>``);
        on miss, falls back to a raw-id lookup. Returns ``None`` when
        the raw-id lookup is ambiguous (more than one provider serves
        the same raw id) so callers can refuse to guess. For the
        bare-id auto-route path see :func:`magos.registry.provider_order.resolve_provider`.
        """
        direct = self.get(model)
        if direct is not None:
            return direct
        providers = self.providers_for_raw_id(model)
        if len(providers) != 1:
            return None
        provider = next(iter(providers))
        return self.get(f"{provider}/{model}")

    def resolve_for_dispatch(self, model: str, provider: str | None) -> str | None:
        """Return the litellm id for ``model``, or ``None`` if unknown.

        Lookup order:
        1. Exact namespaced match (``model`` already is ``<provider>/<raw_id>``).
        2. Provider-scoped raw-id match (``<provider>/<model>``), used when the
           request body carries a bare ``raw_id`` and the rule supplies a provider.

        Namespaced lookup is preferred over the raw scan so an explicit
        ``provider/model`` reference in the request always wins over an accidental
        raw-id collision under a different provider.
        """
        entry = self.get(model)
        if entry is not None:
            return entry.litellm_id
        if provider is not None:
            entry = self.get(f"{provider}/{model}")
            if entry is not None:
                return entry.litellm_id
        return None
