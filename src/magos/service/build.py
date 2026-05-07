"""Factory for ``RequestService``. Shared by every ingress surface."""

from __future__ import annotations

from magos.egress.gateway import (
    CountTokensGateway,
    PassthroughGateway,
    RoutedGateway,
    TranslateGateway,
)
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

    Phase C3 will expand this factory to wire injected ``Compressor`` /
    ``Transform`` collaborators into the routing pipeline.
    """
    router = RuleBasedRouter(
        cfg,
        refresher=refresher,
        registry_settings=registry_cfg.registry,
        providers=registry_cfg.providers,
        pins=registry_cfg.pins,
        provider_order=registry_cfg.provider_order,
    )
    gateway = RoutedGateway(
        passthrough=PassthroughGateway(),
        translate=TranslateGateway(),
        count_tokens=CountTokensGateway(),
    )
    return RequestService(router=router, gateway=gateway)
