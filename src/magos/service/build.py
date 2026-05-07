"""Factory for `RequestService`. Shared by every ingress surface."""

from __future__ import annotations

from magos.dispatch.gateway import (
    CountTokensGateway,
    Gateway,
    MeasuredGateway,
    PassthroughGateway,
    RoutedGateway,
    TracingGateway,
    TranslateGateway,
)
from magos.registry.refresher import Refresher
from magos.registry.schema import RegistryYaml
from magos.routing import RoutingConfig
from magos.routing.engine import MeasuredRouter, Router, RuleBasedRouter
from magos.service.request import RequestService


def build_request_service(
    cfg: RoutingConfig,
    refresher: Refresher | None,
    registry_cfg: RegistryYaml,
    *,
    metrics_enabled: bool = False,
) -> RequestService:
    """Composition root for the application service layer.

    Constructs the router and gateway, optionally wraps them with
    cross-cutting observability decorators, and returns a fully-wired
    `RequestService`.
    """
    router: Router = RuleBasedRouter(
        cfg,
        refresher=refresher,
        registry_settings=registry_cfg.registry,
        providers=registry_cfg.providers,
        pins=registry_cfg.pins,
        provider_order=registry_cfg.provider_order,
    )
    if metrics_enabled:
        router = MeasuredRouter(router)

    gateway: Gateway = RoutedGateway(
        passthrough=PassthroughGateway(),
        translate=TranslateGateway(),
        count_tokens=CountTokensGateway(),
    )
    gateway = TracingGateway(gateway)  # always; no-op when tracing disabled
    if metrics_enabled:
        gateway = MeasuredGateway(gateway)

    return RequestService(router=router, gateway=gateway)
