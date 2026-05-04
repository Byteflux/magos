"""FastAPI app factory.

:func:`create_app` is the canonical entry to build the ASGI app.
Tests pass ``routing`` (and optionally ``registry``) directly to skip
the YAML round-trip; production calls
:func:`magos.config.loader.load_full_config` via the no-arg path.

App-state slots (read by handlers, lifespan, and admin endpoints):

- ``app.state.routing``         — :class:`magos.routing.RoutingConfig`
- ``app.state.registry_config`` — :class:`magos.registry.schema.RegistryYaml`
- ``app.state.refresher``       — :class:`Refresher` or ``None`` when
                                  ``providers:`` is empty

A ``Refresher`` is constructed when the registry block declares any
providers; otherwise the registry feature is dormant and routing
rules behave exactly as before.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from magos import __version__
from magos.config.loader import load_full_config
from magos.config.settings import MagosSettings
from magos.ingress.http.admin import mount_admin_registry_endpoints
from magos.ingress.http.handlers import register_handlers
from magos.ingress.http.lifespan import lifespan
from magos.ingress.http.models import register_models_endpoint
from magos.registry.refresher import Refresher
from magos.registry.schema import RegistryYaml
from magos.routing import RoutingConfig
from magos.telemetry.metrics import mount_metrics_endpoint


def _resolve_models_path(registry_cfg: RegistryYaml, override: str | None) -> Path:
    """Resolve the registry block's ``models_path`` against precedence rules.

    Delegates to :func:`magos.config.loader.resolve_models_path` so server
    boot, CLI ``list --from-disk``, and CLI ``show`` all agree on the
    same file regardless of CWD. ``override`` carries
    ``MAGOS_MODELS_PATH`` (via ``MagosSettings.models_path``) and wins
    over the yaml value. ``models.json`` is server-owned: out-of-
    process readers are fine; the only writer is the Refresher.
    """
    from magos.config.loader import resolve_models_path  # noqa: PLC0415

    return resolve_models_path(registry_cfg, override=override)


def create_app(
    routing: RoutingConfig | None = None,
    *,
    registry: RegistryYaml | None = None,
) -> FastAPI:
    """Build the FastAPI app, loading routing + registry config from disk.

    Tests can pass ``routing`` (and optionally ``registry``) directly to
    skip the YAML round-trip; in that case ``MAGOS_CONFIG_PATH`` is
    ignored. When ``routing`` is omitted, both halves are parsed from
    ``MAGOS_CONFIG_PATH`` via :func:`magos.config.loader.load_full_config`.
    """
    settings = MagosSettings()
    if routing is None:
        full = load_full_config(settings.config_path)
        cfg = full.routing
        registry_cfg = registry if registry is not None else full.registry
    else:
        cfg = routing
        registry_cfg = registry if registry is not None else RegistryYaml()

    app = FastAPI(title="magos", version=__version__, lifespan=lifespan)
    app.state.routing = cfg
    app.state.registry_config = registry_cfg
    app.state.refresher = (
        Refresher(
            registry_cfg,
            _resolve_models_path(registry_cfg, settings.models_path),
        )
        if registry_cfg.providers
        else None
    )

    if settings.metrics_enabled:
        mount_metrics_endpoint(app)
    if app.state.refresher is not None:
        mount_admin_registry_endpoints(app)

    register_handlers(app)
    register_models_endpoint(app)
    return app
