"""Combined config loader: routing rules + registry + ingress from one YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import ValidationError

from magos.config.schema import MagosIngressConfig
from magos.config.settings import magos_home
from magos.registry.discovery.factory import adapter_for
from magos.registry.schema import ProviderConfig, RegistryYaml
from magos.routing.loader import RoutingConfigError
from magos.routing.loader import load_config as load_routing_config
from magos.routing.schema import RoutingConfig


@dataclass(frozen=True, slots=True)
class MagosConfig:
    """Routing rules, registry config, and server config from a single YAML file."""

    routing: RoutingConfig
    registry: RegistryYaml
    ingress: MagosIngressConfig = field(default_factory=MagosIngressConfig)
    source: Path = Path()


def resolve_models_path(registry: RegistryYaml, *, override: str | None = None) -> Path:
    """Resolve the registry's ``models.json`` location to an absolute Path.

    Precedence: ``override`` (``MAGOS_MODELS_PATH``) > yaml ``registry.models_path`` >
    ``"models.json"``. ``~`` expands against OS home; absolute passes through;
    relative anchors to ``$MAGOS_HOME`` (decoupled from yaml parent and CWD).
    """
    raw_str = override or registry.registry.models_path or "models.json"
    if raw_str.startswith("~"):
        return Path(raw_str).expanduser()
    raw = Path(raw_str)
    if raw.is_absolute():
        return raw
    return magos_home() / raw


def load_full_config(path: str | Path) -> MagosConfig:
    """Parse ``path`` into routing + registry + ingress config."""
    routing = load_routing_config(path)
    registry = _parse_registry_block(path)
    registry = _normalize_provider_base_urls(registry)
    ingress = _parse_ingress_block(path)
    return MagosConfig(routing=routing, registry=registry, ingress=ingress, source=Path(path))


def _normalize_provider_base_urls(registry: RegistryYaml) -> RegistryYaml:
    """Fill ``ProviderConfig.base_url`` from the adapter's canonical URL when omitted."""
    if not registry.providers:
        return registry
    updated: dict[str, ProviderConfig] = {}
    for name, cfg in registry.providers.items():
        resolved = cfg
        if resolved.base_url is None:
            adapter = adapter_for(resolved)
            if adapter.default_base_url is not None:
                resolved = resolved.model_copy(update={"base_url": adapter.default_base_url})
        updated[name] = resolved
    return registry.model_copy(update={"providers": updated})


def _parse_registry_block(path: str | Path) -> RegistryYaml:
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    if not isinstance(data, dict):
        raise RoutingConfigError(
            f"{p}: top-level YAML must be a mapping, got {type(data).__name__}"
        )
    # ``extra="forbid"`` on RegistryYaml would reject routing rules; subset first.
    subset = {k: data[k] for k in ("providers", "provider_order", "registry") if k in data}
    try:
        return RegistryYaml.model_validate(subset)
    except ValidationError as exc:
        raise RoutingConfigError(f"{p}: invalid registry config: {exc}") from exc


def _parse_ingress_block(path: str | Path) -> MagosIngressConfig:
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    if not isinstance(data, dict):
        raise RoutingConfigError(
            f"{p}: top-level YAML must be a mapping, got {type(data).__name__}"
        )
    block = data.get("ingress", {})
    try:
        return MagosIngressConfig.model_validate(block)
    except ValidationError as exc:
        raise RoutingConfigError(f"{p}: invalid ingress config: {exc}") from exc
