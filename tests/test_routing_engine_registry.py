"""Tests for registry-driven auto-routing in ``magos.routing.engine``."""

from __future__ import annotations

from magos.registry.models import ModelEntry, RegistryState
from magos.registry.schema import RegistrySettings
from magos.routing.engine import RouteDecision, route
from magos.routing.errors import RouteError
from magos.routing.models import RoutingConfig
from magos.routing.request import RoutedRequest


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
    return RoutedRequest(
        endpoint="/v1/messages",
        headers={},
        body={"model": model},
        raw_body=b"",
    )


def _registry_with(*entries: ModelEntry) -> RegistryState:
    return RegistryState(entries={e.namespaced_id: e for e in entries})


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
