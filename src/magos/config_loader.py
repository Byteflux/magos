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

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import ValidationError

from magos.config import magos_home
from magos.registry.discovery.factory import adapter_for
from magos.registry.schema import ProviderConfig, RegistryYaml
from magos.routing.loader import RoutingConfigError
from magos.routing.loader import load_config as load_routing_config
from magos.routing.models import RoutingConfig
from magos.server_config import MagosServerConfig


@dataclass(frozen=True, slots=True)
class MagosConfig:
    """Routing rules, registry config, and server config from a single YAML file."""

    routing: RoutingConfig
    registry: RegistryYaml
    server: MagosServerConfig = field(default_factory=MagosServerConfig)
    source: Path = Path()  # the yaml file the config was loaded from


def resolve_models_path(registry: RegistryYaml, *, override: str | None = None) -> Path:
    """Resolve the registry's ``models.json`` location to an absolute Path.

    Precedence of which raw string is resolved:

    1. ``override`` (the ``MAGOS_MODELS_PATH`` env var, threaded through
       by callers as ``MagosSettings.models_path``).
    2. ``registry.registry.models_path`` from ``magos.yaml`` when set.
    3. The literal default ``"models.json"``, which after step 4 lands
       at ``$MAGOS_HOME/models.json``.

    Path-string semantics (applied to whichever string wins):

    - ``~``-prefixed → expand against the OS user home directory.
    - Absolute → pass through unchanged.
    - Relative → anchor to ``$MAGOS_HOME`` (default ``~/.magos``), not
      CWD and not the yaml file's parent. ``MAGOS_HOME`` is the magos
      data directory; relative ``models_path`` values are deliberately
      decoupled from where the yaml happens to live.

    ``models.json`` is server-owned: out-of-process readers are fine,
    but the only writer is the running magos server (via the
    Refresher).
    """
    raw_str = override or registry.registry.models_path or "models.json"
    if raw_str.startswith("~"):
        return Path(raw_str).expanduser()
    raw = Path(raw_str)
    if raw.is_absolute():
        return raw
    return magos_home() / raw


def load_full_config(path: str | Path) -> MagosConfig:
    """Parse ``path`` into both routing and registry config.

    The routing half goes through the existing ``load_config`` path
    (post-load validation, regex/jq compilation, passthrough warnings).
    The registry half is purely structural: ``RegistryYaml`` validates
    schema; live discovery and adapter wiring happen in ``Refresher``.
    """
    routing = load_routing_config(path)
    registry = _parse_registry_block(path)
    registry = _normalize_provider_base_urls(registry)
    server = _parse_server_block(path)
    return MagosConfig(routing=routing, registry=registry, server=server, source=Path(path))


def _normalize_provider_base_urls(registry: RegistryYaml) -> RegistryYaml:
    """Fill ``ProviderConfig.base_url`` from the adapter's canonical URL.

    Operators shouldn't have to repeat a vendor's well-known URL in
    ``providers.<name>.base_url`` -- the discovery adapter already knows
    it (Vultr, future custom_openai-routed providers). When operators
    omit ``base_url``, stamp the adapter's ``default_base_url`` so both
    discovery and dispatch see the same URL without per-provider yaml
    boilerplate. Adapters with no canonical host (openai, anthropic,
    openrouter, noop) leave the field None and rely on LiteLLM's
    built-in provider defaults at dispatch time.
    """
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
    # Pull only the registry-related keys so ``extra="forbid"`` on
    # RegistryYaml doesn't reject the routing rules.
    subset = {k: data[k] for k in ("providers", "provider_order", "registry") if k in data}
    try:
        return RegistryYaml.model_validate(subset)
    except ValidationError as exc:
        raise RoutingConfigError(f"{p}: invalid registry config: {exc}") from exc


def _parse_server_block(path: str | Path) -> MagosServerConfig:
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    if not isinstance(data, dict):
        raise RoutingConfigError(
            f"{p}: top-level YAML must be a mapping, got {type(data).__name__}"
        )
    block = data.get("server", {})
    try:
        return MagosServerConfig.model_validate(block)
    except ValidationError as exc:
        raise RoutingConfigError(f"{p}: invalid server config: {exc}") from exc
