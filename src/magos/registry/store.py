"""On-disk persistence for ``RegistryState`` (unversioned JSON).

Parse failure is treated as missing (next refresh rebuilds). Atomic
writes use ``write_temp -> fsync -> os.replace`` on the same filesystem.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import orjson

from magos.registry.state import ModelEntry, RegistryState
from magos.telemetry import get_logger

log = get_logger("magos.registry.store")


def _entry_to_dict(entry: ModelEntry) -> dict[str, Any]:
    return {
        "provider": entry.provider,
        "raw_id": entry.raw_id,
        "litellm_id": entry.litellm_id,
        "context_size": entry.context_size,
        "max_output": entry.max_output,
        "input_cost": entry.input_cost,
        "output_cost": entry.output_cost,
        "cache_read_cost": entry.cache_read_cost,
        "cache_write_cost": entry.cache_write_cost,
        "input_modalities": list(entry.input_modalities),
        "output_modalities": list(entry.output_modalities),
        "deprecated_at": entry.deprecated_at.isoformat() if entry.deprecated_at else None,
        "sources": list(entry.sources),
    }


def _entry_from_dict(data: dict[str, Any]) -> ModelEntry:
    deprecated_raw = data.get("deprecated_at")
    deprecated_at = datetime.fromisoformat(deprecated_raw) if deprecated_raw else None
    input_modalities = data.get("input_modalities") or []
    output_modalities = data.get("output_modalities") or []
    sources = data.get("sources") or []
    return ModelEntry(
        provider=data["provider"],
        raw_id=data["raw_id"],
        litellm_id=data["litellm_id"],
        context_size=data.get("context_size"),
        max_output=data.get("max_output"),
        input_cost=data.get("input_cost"),
        output_cost=data.get("output_cost"),
        cache_read_cost=data.get("cache_read_cost"),
        cache_write_cost=data.get("cache_write_cost"),
        input_modalities=tuple(input_modalities),
        output_modalities=tuple(output_modalities),
        deprecated_at=deprecated_at,
        sources=tuple(sources),
    )


def serialize(state: RegistryState) -> bytes:
    """Render a ``RegistryState`` as canonical JSON bytes."""
    payload: dict[str, Any] = {
        "refreshed_at": {p: ts.isoformat() for p, ts in state.refreshed_at.items()},
        "entries": [_entry_to_dict(e) for e in state.entries.values()],
    }
    return orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)


def deserialize(raw: bytes) -> RegistryState:
    """Parse JSON bytes into a ``RegistryState``; raises on malformed input."""
    payload = orjson.loads(raw)
    refreshed_raw = payload.get("refreshed_at") or {}
    refreshed_at = {p: datetime.fromisoformat(ts) for p, ts in refreshed_raw.items()}
    entries_list = payload.get("entries") or []
    entries: dict[str, ModelEntry] = {}
    for raw_entry in entries_list:
        entry = _entry_from_dict(raw_entry)
        entries[entry.namespaced_id] = entry
    return RegistryState(entries=entries, refreshed_at=refreshed_at)


def load(path: Path) -> RegistryState:
    """Load persisted registry; missing/corrupt yields empty state (regenerable)."""
    if not path.exists():
        return RegistryState()
    try:
        raw = path.read_bytes()
    except OSError as exc:
        log.warning("registry.store.read_failed", path=str(path), error=str(exc))
        return RegistryState()
    try:
        return deserialize(raw)
    except (orjson.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        log.warning("registry.store.parse_failed", path=str(path), error=str(exc))
        return RegistryState()


def save(state: RegistryState, path: Path) -> None:
    """Write ``state`` atomically to ``path`` (write_temp -> fsync -> replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = serialize(state)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # ``os.open`` (rather than ``Path.write_bytes``) so we can ``fsync``
    # the descriptor before the rename; needed for durability across crashes.
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    tmp.replace(path)
