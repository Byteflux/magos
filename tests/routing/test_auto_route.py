"""Tests for ``magos.routing.engine.auto`` registry-driven fallback."""

from __future__ import annotations

from magos.registry.schema import ProviderConfig, RegistrySettings
from magos.registry.state import ModelEntry, RegistryState
from magos.routing.decision import RouteDecision
from magos.routing.engine.auto import (
    _AUTO_ROUTE_RULE_NAME,
    AutoRouter,
    provider_cred_overrides,
)

from ._helpers import make_registry
from ._helpers import make_req as _req


def _router(
    *,
    settings: RegistrySettings | None = None,
    providers: dict[str, ProviderConfig] | None = None,
    pins: dict[str, str] | None = None,
    provider_order: tuple[str, ...] = (),
) -> AutoRouter:
    return AutoRouter(
        registry_settings=settings,
        providers=providers,
        pins=pins,
        provider_order=provider_order,
    )


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


# --- try_route: registry hit ---


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
    decision = _router().try_route(req, registry=registry)
    assert isinstance(decision, RouteDecision)
    assert decision.dispatch_model == e.litellm_id
    assert decision.entry is e


def test_try_auto_route_sets_auto_route_rule_name() -> None:
    e = _entry()
    registry = make_registry(e)
    req = _req(body={"model": e.namespaced_id})
    decision = _router().try_route(req, registry=registry)
    assert isinstance(decision, RouteDecision)
    assert decision.rule.name == _AUTO_ROUTE_RULE_NAME


def test_try_auto_route_stamps_provider_creds_from_provider_cfg() -> None:
    e = _entry(provider="vultr", raw_id="Qwen/Qwen3-30B", litellm_id="custom_openai/Qwen/Qwen3-30B")
    registry = make_registry(e)
    providers = {"vultr": ProviderConfig(api_key_env="VULTR_KEY", base_url="https://vultr.example")}
    req = _req(body={"model": e.namespaced_id})
    decision = _router(providers=providers).try_route(req, registry=registry)
    assert isinstance(decision, RouteDecision)
    assert decision.action.api_key_env == "VULTR_KEY"
    assert decision.action.base_url == "https://vultr.example"


def test_try_auto_route_returns_none_on_registry_miss() -> None:
    registry = RegistryState()
    req = _req(body={"model": "unknown-model"})
    assert _router().try_route(req, registry=registry) is None


def test_try_auto_route_returns_none_on_empty_model() -> None:
    e = _entry()
    registry = make_registry(e)
    req = _req(body={})
    assert _router().try_route(req, registry=registry) is None


# --- try_route: on_unknown_model passthrough ---


def test_try_auto_route_passthrough_on_unknown_when_configured() -> None:
    """``on_unknown_model: passthrough`` yields a translate decision for unknown models."""
    registry = RegistryState()
    settings = RegistrySettings(on_unknown_model="passthrough")
    req = _req(body={"model": "openai/gpt-99"})
    decision = _router(settings=settings).try_route(req, registry=registry)
    assert isinstance(decision, RouteDecision)
    assert decision.dispatch_model == "openai/gpt-99"


def test_try_auto_route_no_passthrough_without_setting() -> None:
    """Without ``on_unknown_model`` setting, miss returns None."""
    registry = RegistryState()
    req = _req(body={"model": "openai/gpt-99"})
    assert _router().try_route(req, registry=registry) is None


def test_try_auto_route_unknown_passthrough_infers_provider_from_slash() -> None:
    registry = RegistryState()
    settings = RegistrySettings(on_unknown_model="passthrough")
    req = _req(body={"model": "myco/my-model"})
    decision = _router(settings=settings).try_route(req, registry=registry)
    assert isinstance(decision, RouteDecision)
    assert decision.action.provider == "myco"


def test_try_auto_route_unknown_passthrough_bare_model_uses_auto_provider() -> None:
    registry = RegistryState()
    settings = RegistrySettings(on_unknown_model="passthrough")
    req = _req(body={"model": "bare-model-no-slash"})
    decision = _router(settings=settings).try_route(req, registry=registry)
    assert isinstance(decision, RouteDecision)
    assert decision.action.provider == "auto"


# --- auto_routed flag on decision ---


def test_try_auto_route_decision_is_auto_routed() -> None:
    e = _entry()
    registry = make_registry(e)
    req = _req(body={"model": e.namespaced_id})
    decision = _router().try_route(req, registry=registry)
    assert isinstance(decision, RouteDecision)
    assert decision.auto_routed is True


# --- try_route: bare-id resolution across providers ---


def test_bare_id_resolves_via_provider_order() -> None:
    """Bare ``model`` with multiple providers picks via ``provider_order``."""
    a = _entry(provider="anthropic", raw_id="claude-x", litellm_id="anthropic/claude-x")
    b = _entry(provider="openrouter", raw_id="claude-x", litellm_id="openrouter/claude-x")
    registry = make_registry(a, b)
    req = _req(body={"model": "claude-x"})
    decision = _router(provider_order=("openrouter", "anthropic")).try_route(req, registry=registry)
    assert isinstance(decision, RouteDecision)
    assert decision.entry is b
    assert decision.action.provider == "openrouter"


def test_bare_id_pin_beats_provider_order() -> None:
    """A pin overrides ``provider_order`` for that raw id."""
    a = _entry(provider="anthropic", raw_id="claude-x", litellm_id="anthropic/claude-x")
    b = _entry(provider="openrouter", raw_id="claude-x", litellm_id="openrouter/claude-x")
    registry = make_registry(a, b)
    req = _req(body={"model": "claude-x"})
    decision = _router(
        pins={"claude-x": "anthropic"},
        provider_order=("openrouter", "anthropic"),
    ).try_route(req, registry=registry)
    assert isinstance(decision, RouteDecision)
    assert decision.entry is a


def test_bare_id_falls_back_to_lex_smallest() -> None:
    """No pin, no ``provider_order`` match: lex-smallest provider wins."""
    a = _entry(provider="zeta", raw_id="claude-x", litellm_id="zeta/claude-x")
    b = _entry(provider="alpha", raw_id="claude-x", litellm_id="alpha/claude-x")
    registry = make_registry(a, b)
    req = _req(body={"model": "claude-x"})
    decision = _router().try_route(req, registry=registry)
    assert isinstance(decision, RouteDecision)
    assert decision.entry is b
    assert decision.action.provider == "alpha"


def test_bare_id_pin_to_absent_provider_is_ignored() -> None:
    """A pin to a provider that doesn't serve this model is ignored."""
    a = _entry(provider="anthropic", raw_id="claude-x", litellm_id="anthropic/claude-x")
    registry = make_registry(a)
    req = _req(body={"model": "claude-x"})
    decision = _router(
        pins={"claude-x": "openrouter"},  # openrouter has no claude-x entry
    ).try_route(req, registry=registry)
    assert isinstance(decision, RouteDecision)
    assert decision.entry is a


def test_bare_id_miss_falls_through_to_unknown_passthrough() -> None:
    """Bare id with no candidates honours ``on_unknown_model: passthrough``."""
    registry = RegistryState()
    settings = RegistrySettings(on_unknown_model="passthrough")
    req = _req(body={"model": "ghost"})
    decision = _router(settings=settings, provider_order=("a",)).try_route(req, registry=registry)
    assert isinstance(decision, RouteDecision)
    assert decision.dispatch_model == "ghost"


def test_namespaced_hit_short_circuits_bare_id_path() -> None:
    """A direct namespaced hit ignores ``provider_order``."""
    a = _entry(provider="anthropic", raw_id="claude-x", litellm_id="anthropic/claude-x")
    b = _entry(provider="openrouter", raw_id="claude-x", litellm_id="openrouter/claude-x")
    registry = make_registry(a, b)
    req = _req(body={"model": "anthropic/claude-x"})
    decision = _router(provider_order=("openrouter",)).try_route(req, registry=registry)
    assert isinstance(decision, RouteDecision)
    assert decision.entry is a
