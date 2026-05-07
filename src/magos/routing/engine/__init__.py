"""`magos.routing.engine`: `Router` ABC + canonical implementations.

Public surface:

- `Router` — ABC.
- `RuleBasedRouter` — declarative rule engine; canonical impl.
- `AutoRouter` — registry-driven fallback.
- `MeasuredRouter` — decorator emitting an OTel counter.
- `route` — free convenience function (constructs a transient
  `RuleBasedRouter` for one-shot use; preferred for tests).
- `apply_pre_transforms` — re-exported for the routing facade.
"""

from __future__ import annotations

from collections.abc import Mapping

from magos.registry.schema import ProviderConfig, RegistrySettings
from magos.registry.state import RegistryState
from magos.routing.decision import RouteDecision
from magos.routing.engine.auto import AutoRouter, provider_cred_overrides
from magos.routing.engine.base import Router
from magos.routing.engine.measured import MeasuredRouter
from magos.routing.engine.rule_based import RuleBasedRouter, _route, apply_pre_transforms
from magos.routing.errors import RouteError
from magos.routing.request import RoutedRequest
from magos.routing.schema import RoutingConfig


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
    """Resolve `req` against `cfg`; convenience wrapper for one-shot use.

    Production code uses `RuleBasedRouter` (long-lived, DI-injected).
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
    "apply_pre_transforms",
    "provider_cred_overrides",
    "route",
]
