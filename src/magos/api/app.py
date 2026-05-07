"""FastAPI app factory. See ``docs/architecture/startup.md``."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from magos import __version__
from magos.api.admin import mount_admin_registry_endpoints
from magos.api.handlers import register_handlers
from magos.api.lifespan import lifespan
from magos.api.models import register_models_endpoint
from magos.config.loader import load_full_config
from magos.config.settings import MagosSettings
from magos.registry.refresher import Refresher
from magos.registry.schema import RegistryYaml
from magos.routing import RoutingConfig
from magos.service import build_request_service
from magos.telemetry.metrics import mount_metrics_endpoint


def _resolve_models_path(registry_cfg: RegistryYaml, override: str | None) -> Path:
    from magos.config.loader import resolve_models_path  # noqa: PLC0415

    return resolve_models_path(registry_cfg, override=override)


def create_app(
    routing: RoutingConfig | None = None,
    *,
    registry: RegistryYaml | None = None,
) -> FastAPI:
    """Build the FastAPI app. ``routing`` passed in skips the yaml load (test seam)."""
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
    app.state.settings = settings
    app.state.refresher = (
        Refresher(
            registry_cfg,
            _resolve_models_path(registry_cfg, settings.models_path),
        )
        if registry_cfg.providers
        else None
    )
    app.state.service = build_request_service(cfg, app.state.refresher, registry_cfg)

    if settings.metrics_enabled:
        mount_metrics_endpoint(app)
    if app.state.refresher is not None:
        mount_admin_registry_endpoints(app)

    register_handlers(app)
    register_models_endpoint(app)
    return app
