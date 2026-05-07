"""``magos.routing.engine``: ``Router`` ABC + canonical implementations.

Public surface:

- :class:`Router` — ABC.
- :class:`RuleBasedRouter` — declarative rule engine; canonical impl.
- :class:`AutoRouter` — registry-driven fallback.
- :class:`MeasuredRouter` — decorator emitting an OTel counter.
- :func:`route` — free convenience function (constructs a transient
  ``RuleBasedRouter`` for one-shot use; preferred for tests).
- :func:`apply_pre_rewrites` — re-exported for the routing facade.
"""

from __future__ import annotations

from collections.abc import Mapping

from magos.registry.schema import ProviderConfig, RegistrySettings
from magos.registry.state import RegistryState
from magos.routing.decision import RouteDecision
from magos.routing.errors import RouteError
from magos.routing.request import RoutedRequest
from magos.routing.schema import RoutingConfig

from .auto import AutoRouter, provider_cred_overrides
from .base import Router
from .measured import MeasuredRouter
from .rule_based import RuleBasedRouter, _route, apply_pre_rewrites


def route(
    req: RoutedRequest,
    cfg: RoutingConfig,
    *,
    registry: RegistryState | None = None,
    registry_settings: RegistrySettings | None = None,
    providers: Mapping[str, ProviderConfig] | None = None,
    pins: Mapping[str, str] | None = None,
    provider_order: tuple[str, ...] = (),
) -> RouteDecision | RouteError:
    """Resolve ``req`` against ``cfg``; convenience wrapper for one-shot use.

    Production code uses :class:`RuleBasedRouter` (long-lived, DI-injected).
    This free function is preferred in tests where constructing a router
    just to call it once is overkill.
    """
    return _route(
        req,
        cfg,
        registry=registry,
        registry_settings=registry_settings,
        providers=providers,
        pins=pins,
        provider_order=provider_order,
    )


__all__ = [
    "AutoRouter",
    "MeasuredRouter",
    "Router",
    "RuleBasedRouter",
    "apply_pre_rewrites",
    "provider_cred_overrides",
    "route",
]
