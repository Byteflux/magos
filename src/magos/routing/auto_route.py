"""Registry-driven auto-routing fallback. See ``docs/registry/auto-routing.md``."""

from __future__ import annotations

from collections.abc import Mapping

from magos.registry.schema import ProviderConfig, RegistrySettings
from magos.registry.state import ModelEntry, RegistryState
from magos.routing.decision import RouteDecision
from magos.routing.request import RoutedRequest
from magos.routing.schema import Action, Rule

_AUTO_ROUTE_RULE_NAME = "auto-route"


def try_auto_route(
    req: RoutedRequest,
    registry: RegistryState,
    settings: RegistrySettings | None,
    providers: Mapping[str, ProviderConfig] | None,
) -> RouteDecision | None:
    """Synthesize a decision from a registry hit, or honour ``on_unknown_model`` on miss."""
    model = str(req.body.get("model", ""))
    if not model:
        return None
    entry = registry.get(model)
    if entry is not None:
        provider_cfg = providers.get(entry.provider) if providers else None
        return _decision_from_entry(req, entry, provider_cfg)
    if settings is not None and settings.on_unknown_model == "passthrough":
        return _decision_for_unknown_passthrough(req, model)
    return None


def _decision_from_entry(
    req: RoutedRequest,
    entry: ModelEntry,
    provider_cfg: ProviderConfig | None,
) -> RouteDecision:
    """Build a synthetic Rule + RouteDecision; stamps provider creds onto the action.

    Without the cred stamp, LiteLLM falls back to per-provider defaults
    (e.g. ``OPENAI_API_KEY`` / ``api.openai.com``) and yields misleading
    401s for ``custom_openai``-style providers. See ``docs/routing/api-keys.md``.
    """
    action_payload: dict[str, str | None] = {
        "provider": entry.provider,
        "mode": "translate",
        **provider_cred_overrides(provider_cfg),
    }
    action = Action.model_validate(action_payload)
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


def provider_cred_overrides(cfg: ProviderConfig | None) -> dict[str, str]:
    """Return the ``api_key_env`` / ``base_url`` subset set on ``cfg`` (empty if None)."""
    if cfg is None:
        return {}
    out: dict[str, str] = {}
    if cfg.api_key_env is not None:
        out["api_key_env"] = cfg.api_key_env
    if cfg.base_url is not None:
        out["base_url"] = cfg.base_url
    return out


def _decision_for_unknown_passthrough(req: RoutedRequest, model: str) -> RouteDecision:
    """Forward an unknown model to LiteLLM; its bundled router resolves or errors."""
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
