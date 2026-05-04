"""Tests for registry-driven auto-routing in ``magos.routing.engine``."""

from __future__ import annotations

from magos.registry.schema import ProviderConfig, RegistrySettings
from magos.registry.state import ModelEntry
from magos.routing.engine import RouteDecision, route
from magos.routing.errors import RouteError
from magos.routing.request import RoutedRequest
from magos.routing.schema import RoutingConfig

from ._helpers import make_registry as _registry_with
from ._helpers import make_req


def _routing_cfg() -> RoutingConfig:
    """Minimal config: one rule that only matches a specific bare model."""
    return RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "name": "explicit",
                    "match": {"model": {"literal": "pinned-model"}},
                    "action": {"provider": "openai", "mode": "translate"},
                }
            ]
        }
    )


def _request(model: str) -> RoutedRequest:
    return make_req(body={"model": model})


def test_explicit_rule_wins_over_registry_match() -> None:
    cfg = _routing_cfg()
    registry = _registry_with(
        ModelEntry(provider="anthropic", raw_id="pinned-model", litellm_id="anthropic/pinned-model")
    )
    result = route(_request("pinned-model"), cfg, registry=registry)
    assert isinstance(result, RouteDecision)
    assert result.action.provider == "openai"  # rule won
    assert result.entry is None


def test_unmatched_model_resolves_via_registry_when_namespaced() -> None:
    cfg = _routing_cfg()
    registry = _registry_with(
        ModelEntry(
            provider="openrouter",
            raw_id="anthropic/claude-sonnet-4-6",
            litellm_id="openrouter/anthropic/claude-sonnet-4-6",
            context_size=200000,
        )
    )
    result = route(_request("openrouter/anthropic/claude-sonnet-4-6"), cfg, registry=registry)
    assert isinstance(result, RouteDecision)
    assert result.auto_routed is True
    assert result.dispatch_model == "openrouter/anthropic/claude-sonnet-4-6"
    assert result.entry is not None
    assert result.entry.context_size == 200000


def test_unmatched_unknown_model_returns_404_by_default() -> None:
    cfg = _routing_cfg()
    registry = _registry_with()
    result = route(_request("not-in-registry"), cfg, registry=registry)
    assert isinstance(result, RouteError)
    assert result.status == 404


def test_unmatched_unknown_model_returns_passthrough_when_configured() -> None:
    cfg = _routing_cfg()
    registry = _registry_with()
    settings = RegistrySettings.model_validate({"on_unknown_model": "passthrough"})
    result = route(
        _request("openai/gpt-4o"),
        cfg,
        registry=registry,
        registry_settings=settings,
    )
    assert isinstance(result, RouteDecision)
    assert result.dispatch_model == "openai/gpt-4o"
    assert result.entry is None
    assert result.action.provider == "openai"  # parsed from prefix


def test_route_without_registry_returns_404_for_unmatched_model() -> None:
    """Backwards-compat: existing callers don't pass registry."""
    cfg = _routing_cfg()
    result = route(_request("anything"), cfg)
    assert isinstance(result, RouteError)
    assert result.status == 404


def test_auto_route_propagates_provider_api_key_env_and_base_url() -> None:
    """Auto-routed action picks up creds from the matching ProviderConfig.

    Without this, an openai-compatible third-party (e.g. Vultr via
    ``custom_openai``) would land in the dispatcher with no api_key/api_base
    and LiteLLM would silently fall back to ``OPENAI_API_KEY`` against
    ``api.openai.com`` -- producing a misleading 401.
    """
    cfg = _routing_cfg()
    registry = _registry_with(
        ModelEntry(
            provider="vultr",
            raw_id="zai-org/GLM-5.1-FP8",
            litellm_id="custom_openai/zai-org/GLM-5.1-FP8",
        )
    )
    providers = {
        "vultr": ProviderConfig.model_validate(
            {
                "api_key_env": "VULTR_API_KEY",
                "base_url": "https://api.vultrinference.com/v1",
            }
        )
    }
    result = route(
        _request("vultr/zai-org/GLM-5.1-FP8"),
        cfg,
        registry=registry,
        providers=providers,
    )
    assert isinstance(result, RouteDecision)
    assert result.auto_routed is True
    assert result.action.api_key_env == "VULTR_API_KEY"
    assert result.action.base_url == "https://api.vultrinference.com/v1"
    assert result.dispatch_model == "custom_openai/zai-org/GLM-5.1-FP8"


def test_auto_route_omits_creds_when_provider_config_missing() -> None:
    """No ProviderConfig for the entry's provider: action stays bare.

    This preserves prior behavior for callers that don't pass providers
    (e.g. older tests). The dispatcher will then fall back to LiteLLM's
    per-provider env-var defaults, which is correct for openai/anthropic
    but wrong for openai-compatible third parties -- those operators
    must declare a ProviderConfig.
    """
    cfg = _routing_cfg()
    registry = _registry_with(
        ModelEntry(
            provider="anthropic",
            raw_id="claude-x",
            litellm_id="anthropic/claude-x",
        )
    )
    result = route(_request("anthropic/claude-x"), cfg, registry=registry)
    assert isinstance(result, RouteDecision)
    assert result.action.api_key_env is None
    assert result.action.base_url is None
