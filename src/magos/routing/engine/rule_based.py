"""``RuleBasedRouter``: declarative rule engine. See ``docs/routing/pipeline.md``."""

from __future__ import annotations

from collections.abc import Mapping

from magos.registry.refresher import Refresher
from magos.registry.schema import ProviderConfig, RegistrySettings
from magos.registry.state import RegistryState
from magos.routing.decision import RouteDecision
from magos.routing.engine.auto import AutoRouter, provider_cred_overrides
from magos.routing.engine.base import Router
from magos.routing.errors import (
    RouteError,
    format_dispatch_error_message,
    format_unmatched_message,
)
from magos.routing.match import matches
from magos.routing.request import RoutedRequest
from magos.routing.rewrites import RewriteError, apply_transforms
from magos.routing.schema import GuardedTransforms, RoutingConfig, Rule, Target


class RuleBasedRouter(Router):
    """Rule-based router. Walks ``cfg.rules``; first match wins; falls through to auto-route.

    Long-lived collaborators are constructor-injected. Dynamic registry
    state is read from ``refresher`` per call (the refresher updates its
    state in the background).
    """

    def __init__(
        self,
        cfg: RoutingConfig,
        *,
        refresher: Refresher | None = None,
        registry_settings: RegistrySettings | None = None,
        providers: Mapping[str, ProviderConfig] | None = None,
        pins: Mapping[str, str] | None = None,
        provider_order: tuple[str, ...] = (),
    ) -> None:
        self._cfg = cfg
        self._refresher = refresher
        self._registry_settings = registry_settings
        self._providers = providers
        self._pins = pins
        self._provider_order = provider_order
        self._auto = AutoRouter(
            registry_settings=registry_settings,
            providers=providers,
            pins=pins,
            provider_order=provider_order,
        )

    def route(self, req: RoutedRequest) -> RouteDecision | RouteError:
        """Resolve ``req`` against ``self._cfg``; first matching rule wins, else auto-route."""
        registry = self._refresher.state if self._refresher is not None else None
        return _route(
            req,
            self._cfg,
            registry=registry,
            registry_settings=self._registry_settings,
            providers=self._providers,
            pins=self._pins,
            provider_order=self._provider_order,
            auto=self._auto,
        )


def apply_pre_transforms(
    req: RoutedRequest,
    cfg: RoutingConfig,
    *,
    registry: RegistryState | None = None,
) -> RoutedRequest:
    """Run global pre-match transforms; guarded entries see prior transforms' effects."""
    out = req
    for entry in cfg.pre_transforms:
        if isinstance(entry, GuardedTransforms):
            if not matches(entry.match, out, registry=registry):
                continue
            out = apply_transforms(out, entry.transforms, registry=registry)
        else:
            out = apply_transforms(out, [entry], registry=registry)
    return out


def _route(
    req: RoutedRequest,
    cfg: RoutingConfig,
    *,
    registry: RegistryState | None,
    registry_settings: RegistrySettings | None,
    providers: Mapping[str, ProviderConfig] | None,
    pins: Mapping[str, str] | None,
    provider_order: tuple[str, ...],
    auto: AutoRouter | None = None,
) -> RouteDecision | RouteError:
    """Core routing pipeline. Shared by ``RuleBasedRouter.route`` and ``route()``."""
    pre_applied = apply_pre_transforms(req, cfg, registry=registry)
    for rule in cfg.rules:
        if not matches(rule.match, pre_applied, registry=registry):
            continue
        try:
            post_applied = apply_transforms(pre_applied, rule.transforms, registry=registry)
        except RewriteError as exc:
            model = str(pre_applied.body.get("model", ""))
            return RouteError(
                status=503,
                code="dispatch_error",
                message=format_dispatch_error_message(str(exc)),
                model=model,
                endpoint=pre_applied.endpoint,
            )
        effective_rule = _fill_target_from_provider_config(rule, providers)
        return RouteDecision(
            rule=effective_rule,
            request=post_applied,
            dispatch_model=_compute_dispatch_model(post_applied, effective_rule.target, registry),
        )

    if registry is not None:
        if auto is None:
            auto = AutoRouter(
                registry_settings=registry_settings,
                providers=providers,
                pins=pins,
                provider_order=provider_order,
            )
        decision = auto.try_route(pre_applied, registry=registry)
        if decision is not None:
            return decision

    model = str(pre_applied.body.get("model", ""))
    return RouteError(
        status=404,
        code="unmatched",
        message=format_unmatched_message(model),
        model=model,
        endpoint=pre_applied.endpoint,
    )


def _fill_target_from_provider_config(
    rule: Rule, providers: Mapping[str, ProviderConfig] | None
) -> Rule:
    """Backfill missing ``api_key_env`` / ``base_url`` from ``providers``.

    Without this fill, LiteLLM falls back to its per-provider defaults
    (e.g. ``OPENAI_API_KEY`` / ``api.openai.com`` for ``custom_openai``-style
    providers like Vultr or hosted vLLM), producing misleading 401s.
    See ``docs/routing/api-keys.md``.
    """
    if providers is None or not rule.target.provider:
        return rule
    overrides = provider_cred_overrides(providers.get(rule.target.provider))
    updates = {key: value for key, value in overrides.items() if getattr(rule.target, key) is None}
    if not updates:
        return rule
    new_target = rule.target.model_copy(update=updates)
    return rule.model_copy(update={"target": new_target})


def _compute_dispatch_model(
    req: RoutedRequest, target: Target, registry: RegistryState | None
) -> str:
    """Return the model id to hand LiteLLM. See ``docs/registry/auto-routing.md``."""
    model = str(req.body.get("model", ""))
    if target.gateway == "passthrough":
        return model
    if registry is not None and model:
        resolved = registry.resolve_for_dispatch(model, target.provider)
        if resolved is not None:
            return resolved
    if "/" in model:
        return model
    return f"{target.provider}/{model}"
