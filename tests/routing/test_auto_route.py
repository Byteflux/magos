"""Tests for ``magos.routing.auto_route`` registry-driven fallback."""

from __future__ import annotations

from magos.registry.schema import ProviderConfig, RegistrySettings
from magos.registry.state import ModelEntry, RegistryState
from magos.routing.auto_route import (
    _AUTO_ROUTE_RULE_NAME,
    provider_cred_overrides,
    try_auto_route,
)
from magos.routing.decision import RouteDecision

from ._helpers import make_registry
from ._helpers import make_req as _req

# --- provider_cred_overrides ---


def test_provider_cred_overrides_none_returns_empty() -> None:
    assert provider_cred_overrides(None) == {}


def test_provider_cred_overrides_includes_api_key_env() -> None:
    cfg = ProviderConfig(api_key_env="MY_KEY")
    assert provider_cred_overrides(cfg) == {"api_key_env": "MY_KEY"}


def test_provider_cred_overrides_includes_base_url() -> None:
    cfg = ProviderConfig(base_url="https://example.com")
    assert provider_cred_overrides(cfg) == {"base_url": "https://example.com"}


def test_provider_cred_overrides_includes_both() -> None:
    cfg = ProviderConfig(api_key_env="MY_KEY", base_url="https://example.com")
    out = provider_cred_overrides(cfg)
    assert out == {"api_key_env": "MY_KEY", "base_url": "https://example.com"}


def test_provider_cred_overrides_omits_unset_fields() -> None:
    """Fields that are None on the config should be absent from the dict."""
    cfg = ProviderConfig()
    assert provider_cred_overrides(cfg) == {}


# --- try_auto_route: registry hit ---


def _entry(
    provider: str = "openrouter",
    raw_id: str = "anthropic/claude-sonnet-4-6",
    litellm_id: str = "openrouter/anthropic/claude-sonnet-4-6",
) -> ModelEntry:
    return ModelEntry(provider=provider, raw_id=raw_id, litellm_id=litellm_id)


def test_try_auto_route_returns_decision_on_registry_hit() -> None:
    e = _entry()
    registry = make_registry(e)
    req = _req(body={"model": e.namespaced_id})
    decision = try_auto_route(req, registry, settings=None, providers=None)
    assert isinstance(decision, RouteDecision)
    assert decision.dispatch_model == e.litellm_id
    assert decision.entry is e


def test_try_auto_route_sets_auto_route_rule_name() -> None:
    e = _entry()
    registry = make_registry(e)
    req = _req(body={"model": e.namespaced_id})
    decision = try_auto_route(req, registry, settings=None, providers=None)
    assert isinstance(decision, RouteDecision)
    assert decision.rule.name == _AUTO_ROUTE_RULE_NAME


def test_try_auto_route_stamps_provider_creds_from_provider_cfg() -> None:
    e = _entry(provider="vultr", raw_id="Qwen/Qwen3-30B", litellm_id="custom_openai/Qwen/Qwen3-30B")
    registry = make_registry(e)
    providers = {"vultr": ProviderConfig(api_key_env="VULTR_KEY", base_url="https://vultr.example")}
    req = _req(body={"model": e.namespaced_id})
    decision = try_auto_route(req, registry, settings=None, providers=providers)
    assert isinstance(decision, RouteDecision)
    assert decision.action.api_key_env == "VULTR_KEY"
    assert decision.action.base_url == "https://vultr.example"


def test_try_auto_route_returns_none_on_registry_miss() -> None:
    registry = RegistryState()
    req = _req(body={"model": "unknown-model"})
    assert try_auto_route(req, registry, settings=None, providers=None) is None


def test_try_auto_route_returns_none_on_empty_model() -> None:
    e = _entry()
    registry = make_registry(e)
    req = _req(body={})
    assert try_auto_route(req, registry, settings=None, providers=None) is None


# --- try_auto_route: on_unknown_model passthrough ---


def test_try_auto_route_passthrough_on_unknown_when_configured() -> None:
    """``on_unknown_model: passthrough`` yields a translate decision for unknown models."""
    registry = RegistryState()
    settings = RegistrySettings(on_unknown_model="passthrough")
    req = _req(body={"model": "openai/gpt-99"})
    decision = try_auto_route(req, registry, settings=settings, providers=None)
    assert isinstance(decision, RouteDecision)
    assert decision.dispatch_model == "openai/gpt-99"


def test_try_auto_route_no_passthrough_without_setting() -> None:
    """Without ``on_unknown_model`` setting, miss returns None."""
    registry = RegistryState()
    req = _req(body={"model": "openai/gpt-99"})
    assert try_auto_route(req, registry, settings=None, providers=None) is None


def test_try_auto_route_unknown_passthrough_infers_provider_from_slash() -> None:
    registry = RegistryState()
    settings = RegistrySettings(on_unknown_model="passthrough")
    req = _req(body={"model": "myco/my-model"})
    decision = try_auto_route(req, registry, settings=settings, providers=None)
    assert isinstance(decision, RouteDecision)
    assert decision.action.provider == "myco"


def test_try_auto_route_unknown_passthrough_bare_model_uses_auto_provider() -> None:
    registry = RegistryState()
    settings = RegistrySettings(on_unknown_model="passthrough")
    req = _req(body={"model": "bare-model-no-slash"})
    decision = try_auto_route(req, registry, settings=settings, providers=None)
    assert isinstance(decision, RouteDecision)
    assert decision.action.provider == "auto"


# --- auto_routed flag on decision ---


def test_try_auto_route_decision_is_auto_routed() -> None:
    e = _entry()
    registry = make_registry(e)
    req = _req(body={"model": e.namespaced_id})
    decision = try_auto_route(req, registry, settings=None, providers=None)
    assert isinstance(decision, RouteDecision)
    assert decision.auto_routed is True
