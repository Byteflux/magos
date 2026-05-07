"""Factory for ``RequestService``. Shared by every ingress surface."""

from __future__ import annotations

from magos.registry.refresher import Refresher
from magos.registry.schema import RegistryYaml
from magos.routing import RoutingConfig
from magos.routing.engine import RuleBasedRouter

from .request import RequestService


def build_request_service(
    cfg: RoutingConfig,
    refresher: Refresher | None,
    registry_cfg: RegistryYaml,
) -> RequestService:
    """Construct the ``RequestService`` from long-lived collaborators.

    Phases C2, C3 will expand this factory to wire injected ``Gateway``
    and ``Compressor`` collaborators.
    """
    router = RuleBasedRouter(
        cfg,
        refresher=refresher,
        registry_settings=registry_cfg.registry,
        providers=registry_cfg.providers,
        pins=registry_cfg.pins,
        provider_order=registry_cfg.provider_order,
    )
    return RequestService(router=router)
