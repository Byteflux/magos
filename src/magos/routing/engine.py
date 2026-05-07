"""Routing pipeline: pre-rewrites, match, post-rewrites, decision. See ``docs/routing/pipeline.md``."""

from __future__ import annotations

from collections.abc import Mapping

from magos.registry.schema import ProviderConfig, RegistrySettings
from magos.registry.state import RegistryState
from magos.routing.auto_route import provider_cred_overrides, try_auto_route
from magos.routing.decision import RouteDecision
from magos.routing.errors import (
    RouteError,
    format_dispatch_error_message,
    format_unmatched_message,
)
from magos.routing.matchers import matches
from magos.routing.request import RoutedRequest
from magos.routing.rewrites import RewriteError, apply_rewrites
from magos.routing.schema import Action, GuardedRewrites, RoutingConfig, Rule


def apply_pre_rewrites(
    req: RoutedRequest,
    cfg: RoutingConfig,
    *,
    registry: RegistryState | None = None,
) -> RoutedRequest:
    """Run global pre-match rewrites; guarded entries see prior rewrites' effects."""
    out = req
    for entry in cfg.pre_rewrites:
        if isinstance(entry, GuardedRewrites):
            if not matches(entry.match, out, registry=registry):
                continue
            out = apply_rewrites(out, entry.rewrites, registry=registry)
        else:
            out = apply_rewrites(out, [entry], registry=registry)
    return out


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
    """Resolve ``req`` against ``cfg``; first matching rule wins, else auto-route.

    See ``docs/routing/pipeline.md`` and ``docs/registry/auto-routing.md``.
    """
    pre_applied = apply_pre_rewrites(req, cfg, registry=registry)
    for rule in cfg.rules:
        if not matches(rule.match, pre_applied, registry=registry):
            continue
        try:
            post_applied = apply_rewrites(pre_applied, rule.rewrites, registry=registry)
        except RewriteError as exc:
            model = str(pre_applied.body.get("model", ""))
            return RouteError(
                status=503,
                code="dispatch_error",
                message=format_dispatch_error_message(str(exc)),
                model=model,
                endpoint=pre_applied.endpoint,
            )
        effective_rule = _fill_action_from_provider_config(rule, providers)
        return RouteDecision(
            rule=effective_rule,
            request=post_applied,
            dispatch_model=_compute_dispatch_model(post_applied, effective_rule.action, registry),
        )

    if registry is not None:
        auto = try_auto_route(
            pre_applied,
            registry,
            registry_settings,
            providers,
            pins=pins,
            provider_order=provider_order,
        )
        if auto is not None:
            return auto

    model = str(pre_applied.body.get("model", ""))
    return RouteError(
        status=404,
        code="unmatched",
        message=format_unmatched_message(model),
        model=model,
        endpoint=pre_applied.endpoint,
    )


def _fill_action_from_provider_config(
    rule: Rule, providers: Mapping[str, ProviderConfig] | None
) -> Rule:
    """Backfill missing ``api_key_env`` / ``base_url`` from ``providers``.

    Without this fill, LiteLLM falls back to its per-provider defaults
    (e.g. ``OPENAI_API_KEY`` / ``api.openai.com`` for ``custom_openai``-style
    providers like Vultr or hosted vLLM), producing misleading 401s.
    See ``docs/routing/api-keys.md``.
    """
    if providers is None or not rule.action.provider:
        return rule
    overrides = provider_cred_overrides(providers.get(rule.action.provider))
    updates = {key: value for key, value in overrides.items() if getattr(rule.action, key) is None}
    if not updates:
        return rule
    new_action = rule.action.model_copy(update=updates)
    return rule.model_copy(update={"action": new_action})


def _compute_dispatch_model(
    req: RoutedRequest, action: Action, registry: RegistryState | None
) -> str:
    """Return the model id to hand LiteLLM. See ``docs/registry/auto-routing.md``."""
    model = str(req.body.get("model", ""))
    if action.mode == "passthrough":
        return model
    if registry is not None and model:
        resolved = registry.resolve_for_dispatch(model, action.provider)
        if resolved is not None:
            return resolved
    if "/" in model:
        return model
    return f"{action.provider}/{model}"
