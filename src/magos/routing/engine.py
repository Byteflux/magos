"""Routing pipeline: pre-rewrites, match, post-rewrites, decision.

``route()`` is the single public entry point. It returns either a
``RouteDecision`` describing how the dispatcher should handle the request,
or a ``RouteError`` carrying the status code and message the server should
serialise into the per-endpoint error envelope.

The engine is deliberately stateless: every call recompiles regex/jq
artifacts via the matcher and rewrite layers. The stdlib's ``re`` cache
covers regex; ``jq.compile`` is fast enough at the current rule counts
that adding our own cache would be premature.

When a registry is wired in, requests that no explicit rule matches fall
through to registry-driven auto-routing: an exact ``<provider>/<raw_id>``
lookup against ``RegistryState.entries``. The registry never overrides an
explicit rule (rules win); it only catches what rules miss. ``on_unknown_model``
controls what happens when the registry also misses (404 default,
passthrough opt-in).
"""

from __future__ import annotations

from dataclasses import dataclass

from magos.registry.models import ModelEntry, RegistryState
from magos.registry.schema import RegistrySettings
from magos.routing.errors import (
    RouteError,
    format_dispatch_error_message,
    format_unmatched_message,
)
from magos.routing.matchers import matches
from magos.routing.models import Action, GuardedRewrites, RoutingConfig, Rule
from magos.routing.request import RoutedRequest
from magos.routing.rewrites import RewriteError, apply_rewrites

_AUTO_ROUTE_RULE_NAME = "auto-route"


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Outcome of a successful route lookup, consumed by the dispatcher.

    ``entry`` is the registry record that produced an auto-routed
    decision (None for explicit-rule decisions). Downstream code can
    use it to read context_size, modalities, etc. without re-querying
    the registry.
    """

    rule: Rule
    request: RoutedRequest
    dispatch_model: str
    entry: ModelEntry | None = None

    @property
    def action(self) -> Action:
        return self.rule.action

    @property
    def auto_routed(self) -> bool:
        return self.entry is not None

    def rule_label(self, idx: int | None = None) -> str:
        """Stable human-readable identifier for logs."""
        if self.rule.name is not None:
            return self.rule.name
        if idx is not None:
            return f"rule[{idx}]"
        return "rule[?]"


def apply_pre_rewrites(
    req: RoutedRequest,
    cfg: RoutingConfig,
    *,
    registry: RegistryState | None = None,
) -> RoutedRequest:
    """Run the global pre-match rewrites against ``req``.

    Each entry is either a bare ``Rewrite`` (always applied) or a
    ``GuardedRewrites`` group whose inner rewrites apply only when its
    ``match`` evaluates true against the request as it stands at that
    point in the chain. Earlier guarded entries see the original
    request; later entries see the cumulative effect of the prior ones.
    """
    out = req
    for entry in cfg.pre_rewrites:
        if isinstance(entry, GuardedRewrites):
            if not matches(entry.match, out, registry=registry):
                continue
            out = apply_rewrites(out, entry.rewrites, registry=registry)
        else:
            out = apply_rewrites(out, [entry], registry=registry)
    return out


def apply_post_rewrites(
    req: RoutedRequest,
    rule: Rule,
    *,
    registry: RegistryState | None = None,
) -> RoutedRequest:
    """Run the matched rule's per-rule rewrites against ``req``."""
    return apply_rewrites(req, rule.rewrites, registry=registry)


def route(
    req: RoutedRequest,
    cfg: RoutingConfig,
    *,
    registry: RegistryState | None = None,
    registry_settings: RegistrySettings | None = None,
) -> RouteDecision | RouteError:
    """Resolve ``req`` against ``cfg``; first matching rule wins.

    On rules-loop fall-through, attempt registry auto-routing if a
    ``RegistryState`` is supplied. ``registry_settings`` controls the
    miss behavior (``on_unknown_model`` field); when omitted, defaults
    to error-on-unknown.
    """
    pre_applied = apply_pre_rewrites(req, cfg, registry=registry)
    for rule in cfg.rules:
        if not matches(rule.match, pre_applied, registry=registry):
            continue
        try:
            post_applied = apply_post_rewrites(pre_applied, rule, registry=registry)
        except RewriteError as exc:
            model = str(pre_applied.body.get("model", ""))
            return RouteError(
                status=503,
                code="dispatch_error",
                message=format_dispatch_error_message(str(exc)),
                model=model,
                endpoint=pre_applied.endpoint,
            )
        return RouteDecision(
            rule=rule,
            request=post_applied,
            dispatch_model=_compute_dispatch_model(post_applied, rule.action),
        )

    if registry is not None:
        auto = _try_auto_route(pre_applied, registry, registry_settings)
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


def _try_auto_route(
    req: RoutedRequest,
    registry: RegistryState,
    settings: RegistrySettings | None,
) -> RouteDecision | None:
    """Look up ``req.body['model']`` in the registry by exact namespaced id.

    Returns a synthesized ``RouteDecision`` on hit. On miss, consults
    ``settings.on_unknown_model``: ``"passthrough"`` returns a best-effort
    decision that hands the raw model string to LiteLLM (which resolves
    via its bundled registry on names like ``openai/gpt-4o``); ``"error"``
    returns ``None`` so the caller emits the standard 404.
    """
    model = str(req.body.get("model", ""))
    if not model:
        return None
    entry = registry.get(model)
    if entry is not None:
        return _decision_from_entry(req, entry)
    if settings is not None and settings.on_unknown_model == "passthrough":
        return _decision_for_unknown_passthrough(req, model)
    return None


def _decision_from_entry(req: RoutedRequest, entry: ModelEntry) -> RouteDecision:
    """Build a synthetic Rule + RouteDecision around a registry entry."""
    action = Action.model_validate({"provider": entry.provider, "mode": "translate"})
    rule = Rule.model_validate(
        {
            "name": _AUTO_ROUTE_RULE_NAME,
            "match": {"model": {"literal": entry.raw_id}},
            "action": action.model_dump(),
        }
    )
    return RouteDecision(
        rule=rule,
        request=req,
        dispatch_model=entry.litellm_id,
        entry=entry,
    )


def _decision_for_unknown_passthrough(req: RoutedRequest, model: str) -> RouteDecision:
    """Build a synthetic decision that forwards an unknown model to LiteLLM.

    Used when ``on_unknown_model: passthrough``. The dispatch model is
    the raw inbound id; LiteLLM's bundled provider router resolves it
    if it can, otherwise the provider replies with its own error.
    """
    provider = model.split("/", 1)[0] if "/" in model else "auto"
    action = Action.model_validate({"provider": provider, "mode": "translate"})
    rule = Rule.model_validate(
        {
            "name": "auto-passthrough",
            "match": {"model": {"literal": model}},
            "action": action.model_dump(),
        }
    )
    return RouteDecision(rule=rule, request=req, dispatch_model=model)


def _compute_dispatch_model(req: RoutedRequest, action: Action) -> str:
    """Return the model identifier the dispatcher should hand to litellm.

    Translate mode prepends ``<provider>/`` when the body's model lacks a
    provider prefix; LiteLLM rejects bare names. Passthrough does not go
    through LiteLLM, so the bare model is preserved for logging only.
    """
    model = str(req.body.get("model", ""))
    if action.mode == "passthrough":
        return model
    if "/" in model:
        return model
    return f"{action.provider}/{model}"
