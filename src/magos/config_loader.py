"""Combined config loader: routing rules + registry blocks from one YAML.

``magos.yaml`` historically held only the routing rules (``RoutingConfig``).
The registry batch adds three new top-level keys — ``providers``,
``provider_order``, ``registry`` — that ``RoutingConfig`` doesn't know
about. Rather than fold them into ``RoutingConfig`` (which would mix
concerns and confuse round-trips), this module parses the same file
twice into two pydantic schemas and returns a single ``MagosConfig``
container.

``load_routing_config`` (in ``routing.loader``) keeps its narrow contract
for callers that don't care about the registry. Server lifespan and CLI
use ``load_full_config`` to get both halves.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import ValidationError

from magos.registry.schema import RegistryYaml
from magos.routing.loader import RoutingConfigError
from magos.routing.loader import load_config as load_routing_config
from magos.routing.models import RoutingConfig


@dataclass(frozen=True, slots=True)
class MagosConfig:
    """Routing rules and registry config parsed from a single YAML file."""

    routing: RoutingConfig
    registry: RegistryYaml
    source: Path = Path()  # the yaml file the config was loaded from


def resolve_models_path(config_path: str | Path, registry: RegistryYaml) -> Path:
    """Resolve ``registry.models_path`` relative to the config file's parent.

    Absolute paths pass through unchanged. Relative paths anchor to the
    yaml file's directory so server boot, CLI list, and CLI refresh all
    agree on which file is in play regardless of CWD. ``models.json`` is
    server-owned: out-of-process readers are fine, but the only writer
    is the running magos server (via the Refresher).
    """
    raw = Path(registry.registry.models_path)
    if raw.is_absolute():
        return raw
    return Path(config_path).resolve().parent / raw


def load_full_config(path: str | Path) -> MagosConfig:
    """Parse ``path`` into both routing and registry config.

    The routing half goes through the existing ``load_config`` path
    (post-load validation, regex/jq compilation, passthrough warnings).
    The registry half is purely structural: ``RegistryYaml`` validates
    schema; live discovery and adapter wiring happen in ``Refresher``.
    """
    routing = load_routing_config(path)
    registry = _parse_registry_block(path)
    return MagosConfig(routing=routing, registry=registry, source=Path(path))


def _parse_registry_block(path: str | Path) -> RegistryYaml:
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    if not isinstance(data, dict):
        raise RoutingConfigError(
            f"{p}: top-level YAML must be a mapping, got {type(data).__name__}"
        )
    # Pull only the registry-related keys so ``extra="forbid"`` on
    # RegistryYaml doesn't reject the routing rules.
    subset = {k: data[k] for k in ("providers", "provider_order", "registry") if k in data}
    try:
        return RegistryYaml.model_validate(subset)
    except ValidationError as exc:
        raise RoutingConfigError(f"{p}: invalid registry config: {exc}") from exc
