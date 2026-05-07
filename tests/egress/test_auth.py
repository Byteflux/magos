"""Tests for ``magos.egress.auth`` API-key resolution + header injection."""

from __future__ import annotations

import pytest

from magos.egress.auth import maybe_inject_api_key
from magos.routing.schema import Target


@pytest.mark.unit
def test_inject_api_key_defaults_to_bearer_for_non_anthropic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """openai/openrouter/vultr providers get ``Authorization: Bearer`` by default."""
    monkeypatch.setenv("VULTR_API_KEY", "vk-test")
    action = Target.model_validate(
        {
            "provider": "vultr",
            "gateway": "passthrough",
            "base_url": "https://api.vultrinference.com",
            "api_key_env": "VULTR_API_KEY",
        }
    )
    out = maybe_inject_api_key({}, action)
    assert out == {"authorization": "Bearer vk-test"}
    assert "x-api-key" not in out


@pytest.mark.unit
def test_inject_api_key_anthropic_default_uses_x_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anthropic provider keeps the official ``x-api-key`` header shape."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    action = Target.model_validate(
        {
            "provider": "anthropic",
            "gateway": "passthrough",
            "base_url": "https://api.anthropic.com",
            "api_key_env": "ANTHROPIC_API_KEY",
        }
    )
    out = maybe_inject_api_key({}, action)
    assert out == {"x-api-key": "sk-ant-test"}
    assert "authorization" not in out


@pytest.mark.unit
def test_inject_api_key_anthropic_oauth_token_uses_bearer_plus_beta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claude-Code OAuth tokens force Bearer + ``anthropic-beta: oauth-...``.

    api.anthropic.com rejects ``sk-ant-oat...`` tokens on the ``x-api-key``
    header with 401 ``invalid x-api-key``; the only accepted shape is the
    OAuth one. The detection must override both the per-provider default
    and any explicit ``auth_header`` setting on the rule, since neither
    alternative will authenticate.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-oat01-deadbeef")

    default_shape = Target.model_validate(
        {
            "provider": "anthropic",
            "gateway": "passthrough",
            "base_url": "https://api.anthropic.com",
            "api_key_env": "ANTHROPIC_API_KEY",
        }
    )
    assert maybe_inject_api_key({}, default_shape) == {
        "authorization": "Bearer sk-ant-oat01-deadbeef",
        "anthropic-beta": "oauth-2025-04-20",
    }

    # Explicit x-api-key override is intentionally ignored for OAuth tokens.
    explicit_xapikey = Target.model_validate(
        {
            "provider": "anthropic",
            "gateway": "passthrough",
            "base_url": "https://api.anthropic.com",
            "api_key_env": "ANTHROPIC_API_KEY",
            "auth_header": "x-api-key",
        }
    )
    assert maybe_inject_api_key({}, explicit_xapikey) == {
        "authorization": "Bearer sk-ant-oat01-deadbeef",
        "anthropic-beta": "oauth-2025-04-20",
    }


@pytest.mark.unit
def test_inject_api_key_explicit_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """``target.auth_header`` overrides the per-provider default both ways."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    anthropic_bearer = Target.model_validate(
        {
            "provider": "anthropic",
            "gateway": "passthrough",
            "base_url": "https://api.anthropic.com",
            "api_key_env": "ANTHROPIC_API_KEY",
            "auth_header": "bearer",
        }
    )
    assert maybe_inject_api_key({}, anthropic_bearer) == {"authorization": "Bearer sk-ant-test"}

    openai_xapikey = Target.model_validate(
        {
            "provider": "openai",
            "gateway": "passthrough",
            "base_url": "https://api.openai.com",
            "api_key_env": "OPENAI_API_KEY",
            "auth_header": "x-api-key",
        }
    )
    assert maybe_inject_api_key({}, openai_xapikey) == {"x-api-key": "sk-test"}


@pytest.mark.unit
def test_inject_api_key_skips_when_inbound_auth_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Client-supplied auth always wins; injection never overwrites it."""
    monkeypatch.setenv("VULTR_API_KEY", "vk-test")
    action = Target.model_validate(
        {
            "provider": "vultr",
            "gateway": "passthrough",
            "base_url": "https://api.vultrinference.com",
            "api_key_env": "VULTR_API_KEY",
        }
    )
    inbound_auth = {"authorization": "Bearer client-supplied"}
    assert maybe_inject_api_key(inbound_auth, action) == inbound_auth

    inbound_xapikey = {"x-api-key": "client-supplied"}
    assert maybe_inject_api_key(inbound_xapikey, action) == inbound_xapikey


@pytest.mark.unit
def test_inject_api_key_noop_in_translate_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Translate mode never injects; api_key plumbing happens via litellm kwargs."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    action = Target.model_validate(
        {
            "provider": "openai",
            "gateway": "translate",
            "api_key_env": "OPENAI_API_KEY",
        }
    )
    assert maybe_inject_api_key({}, action) == {}
