"""Pure pipeline functions: merge, diff, and override conversion.

These compose ``merge_entries`` (single-entry field precedence from
``magos.registry.merge``) with discovery results, overrides, and LiteLLM
lookups into the higher-level per-provider operations used by the refresher.
No I/O, no logging, no async.

See ``docs/registry/overview.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from magos.registry.discovery.base import DiscoveryResult
from magos.registry.litellm_lookup import GetModelInfoFn, PartialEntry, lookup
from magos.registry.merge import merge
from magos.registry.schema import ModelOverride, ProviderConfig
from magos.registry.state import ModelEntry


@dataclass(frozen=True, slots=True)
class ProviderDiff:
    """Per-provider deltas produced by ``diff_provider`` for observability."""

    total: int
    added: int
    deprecated: int
    pruned: int


def override_to_partial(override: ModelOverride | None) -> PartialEntry | None:
    """Convert a ``ModelOverride`` config entry to a ``PartialEntry`` for merge.

    Returns ``None`` when ``override`` is ``None`` so callers can pass the
    result directly to ``merge`` without extra None-checks.
    """
    if override is None:
        return None
    return PartialEntry(
        litellm_id=override.litellm_id,
        context_size=override.context_size,
        max_output=override.max_output,
        input_cost=override.input_cost,
        output_cost=override.output_cost,
        cache_read_cost=override.cache_read_cost,
        cache_write_cost=override.cache_write_cost,
        input_modalities=override.input_modalities,
        output_modalities=override.output_modalities,
    )


def merge_provider(
    provider_name: str,
    cfg: ProviderConfig,
    result: DiscoveryResult,
    litellm_lookup: GetModelInfoFn | None = None,
) -> dict[str, ModelEntry]:
    """Build fresh entries for ``provider_name``: discovered models +
    their overrides + litellm fallback, plus override-only entries
    (those not seen in discovery) synthesised manually.
    """
    fresh: dict[str, ModelEntry] = {}
    seen_raw_ids: set[str] = set()
    for discovered in result.models:
        raw_id = discovered.raw_id
        seen_raw_ids.add(raw_id)
        override_partial = override_to_partial(cfg.models.get(raw_id))
        litellm_fallback = _litellm_partial(
            _effective_litellm_id(discovered.litellm_id, cfg.models.get(raw_id)),
            litellm_lookup,
        )
        entry = merge(
            provider=provider_name,
            raw_id=raw_id,
            default_litellm_id=discovered.litellm_id,
            override=override_partial,
            discovered=discovered.partial,
            litellm_fallback=litellm_fallback,
        )
        fresh[entry.namespaced_id] = entry

    for raw_id, override in cfg.models.items():
        if raw_id in seen_raw_ids:
            continue
        default_litellm_id = override.litellm_id or _default_manual_litellm_id(
            provider_name, cfg, raw_id
        )
        override_partial = override_to_partial(override)
        litellm_fallback = _litellm_partial(default_litellm_id, litellm_lookup)
        entry = merge(
            provider=provider_name,
            raw_id=raw_id,
            default_litellm_id=default_litellm_id,
            override=override_partial,
            discovered=None,
            litellm_fallback=litellm_fallback,
        )
        fresh[entry.namespaced_id] = entry
    return fresh


def diff_provider(
    provider: str,
    prev_entries: Mapping[str, ModelEntry],
    next_entries: Mapping[str, ModelEntry],
) -> ProviderDiff:
    """Per-provider deltas: added/deprecated/pruned plus post-refresh total
    (including still-marked deprecated entries within the grace window).
    """
    prev_for_provider = {k: e for k, e in prev_entries.items() if e.provider == provider}
    next_for_provider = {k: e for k, e in next_entries.items() if e.provider == provider}
    added = sum(1 for k in next_for_provider if k not in prev_for_provider)
    deprecated = sum(
        1
        for k, entry in next_for_provider.items()
        if entry.deprecated_at is not None
        and (k not in prev_for_provider or prev_for_provider[k].deprecated_at is None)
    )
    pruned = sum(1 for k in prev_for_provider if k not in next_for_provider)
    return ProviderDiff(
        total=len(next_for_provider),
        added=added,
        deprecated=deprecated,
        pruned=pruned,
    )


def _litellm_partial(litellm_id: str, get_info: GetModelInfoFn | None) -> PartialEntry:
    if get_info is None:
        return lookup(litellm_id)
    return lookup(litellm_id, get_info=get_info)


def _effective_litellm_id(default: str, override: ModelOverride | None) -> str:
    if override is None or override.litellm_id is None:
        return default
    return override.litellm_id


def _default_manual_litellm_id(provider_name: str, cfg: ProviderConfig, raw_id: str) -> str:
    """``<litellm_provider or provider_name>/<raw_id>`` for manual-only entries."""
    prefix = cfg.litellm_provider or provider_name
    return f"{prefix}/{raw_id}"
