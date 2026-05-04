"""Registry-driven auto-routing fallback.

When no explicit rule in ``cfg.rules`` matches, ``route()`` falls through
to here. We do an exact ``<provider>/<raw_id>`` lookup against
``RegistryState.entries`` and synthesize a ``RouteDecision`` from the
matching ``ModelEntry``. The registry never overrides explicit rules; it
only catches what they miss.

``on_unknown_model`` controls what happens on registry miss:
``"passthrough"`` hands the raw model string to LiteLLM (which has its
own bundled provider router for names like ``openai/gpt-4o``);
``"error"`` returns ``None`` so the caller emits the standard 404.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from magos.registry.schema import ProviderConfig, RegistrySettings
from magos.registry.state import ModelEntry, RegistryState
from magos.routing.request import RoutedRequest
from magos.routing.schema import Action, Rule

if TYPE_CHECKING:
    from magos.routing.engine import RouteDecision

_AUTO_ROUTE_RULE_NAME = "auto-route"


def try_auto_route(
    req: RoutedRequest,
    registry: RegistryState,
    settings: RegistrySettings | None,
    providers: Mapping[str, ProviderConfig] | None,
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
    """Build a synthetic Rule + RouteDecision around a registry entry.

    Stamps the matching ``ProviderConfig``'s ``api_key_env`` and
    ``base_url`` onto the synthesized translate-mode ``Action``. Without
    this, LiteLLM is invoked with no api_key/api_base and silently falls
    back to its per-provider defaults (e.g. ``OPENAI_API_KEY`` against
    ``api.openai.com``), producing misleading 401s for openai-compatible
    third parties (Vultr, hosted vLLM, etc.) that route through the
    generic ``custom_openai`` provider.
    """
    from magos.routing.engine import RouteDecision  # noqa: PLC0415

    action_payload: dict[str, str | None] = {
        "provider": entry.provider,
        "mode": "translate",
    }
    if provider_cfg is not None:
        if provider_cfg.api_key_env is not None:
            action_payload["api_key_env"] = provider_cfg.api_key_env
        if provider_cfg.base_url is not None:
            action_payload["base_url"] = provider_cfg.base_url
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


def _decision_for_unknown_passthrough(req: RoutedRequest, model: str) -> RouteDecision:
    """Build a synthetic decision that forwards an unknown model to LiteLLM.

    Used when ``on_unknown_model: passthrough``. The dispatch model is
    the raw inbound id; LiteLLM's bundled provider router resolves it
    if it can, otherwise the provider replies with its own error.
    """
    from magos.routing.engine import RouteDecision  # noqa: PLC0415

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
