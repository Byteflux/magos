"""Routing pipeline: pre-rewrites, match, post-rewrites, decision.

``route()`` is the single public entry point. It returns either a
``RouteDecision`` describing how the dispatcher should handle the request,
or a ``RouteError`` carrying the status code and message the server should
serialise into the per-endpoint error envelope.

The engine is deliberately stateless: every call recompiles regex/jq
artifacts via the matcher and rewrite layers. A future optimisation can add
a per-rule compiled-artifact cache here without changing the public API,
keyed by rule identity to avoid fighting pydantic's frozen models.
"""

from __future__ import annotations

from dataclasses import dataclass

from magos.routing.errors import (
    RouteError,
    format_dispatch_error_message,
    format_unmatched_message,
)
from magos.routing.matchers import matches
from magos.routing.models import Action, RoutingConfig, Rule
from magos.routing.request import RoutedRequest
from magos.routing.rewrites import RewriteError, apply_rewrites


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Outcome of a successful route lookup, consumed by the dispatcher."""

    rule: Rule
    request: RoutedRequest
    dispatch_model: str

    @property
    def action(self) -> Action:
        return self.rule.action

    def rule_label(self, idx: int | None = None) -> str:
        """Stable human-readable identifier for logs."""
        if self.rule.name is not None:
            return self.rule.name
        if idx is not None:
            return f"rule[{idx}]"
        return "rule[?]"


def apply_pre_rewrites(req: RoutedRequest, cfg: RoutingConfig) -> RoutedRequest:
    """Run the global pre-match rewrites against ``req``."""
    return apply_rewrites(req, cfg.pre_rewrites)


def apply_post_rewrites(req: RoutedRequest, rule: Rule) -> RoutedRequest:
    """Run the matched rule's per-rule rewrites against ``req``."""
    return apply_rewrites(req, rule.rewrites)


def route(req: RoutedRequest, cfg: RoutingConfig) -> RouteDecision | RouteError:
    """Resolve ``req`` against ``cfg``; first matching rule wins."""
    pre_applied = apply_pre_rewrites(req, cfg)
    for rule in cfg.rules:
        if not matches(rule.match, pre_applied):
            continue
        try:
            post_applied = apply_post_rewrites(pre_applied, rule)
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
    model = str(pre_applied.body.get("model", ""))
    return RouteError(
        status=404,
        code="unmatched",
        message=format_unmatched_message(model),
        model=model,
        endpoint=pre_applied.endpoint,
    )


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
