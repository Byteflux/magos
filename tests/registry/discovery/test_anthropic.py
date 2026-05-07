"""`AnthropicAdapter` discovery tests (incl. OAuth-token auth shape)."""

from __future__ import annotations

import httpx
import pytest

from magos.registry.discovery.anthropic import AnthropicAdapter
from magos.registry.discovery.base import DiscoveryError, DiscoveryResult
from magos.registry.schema import ProviderConfig
from tests.registry.discovery._helpers import ok, run_discover


def _run(cfg: ProviderConfig, transport: httpx.MockTransport) -> DiscoveryResult:
    return run_discover(AnthropicAdapter(), "anthropic", cfg, transport)


def test_anthropic_adapter_extracts_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_KEY", "sk-ant-test")
    cfg = ProviderConfig.model_validate({"api_key_env": "ANTHROPIC_KEY"})
    transport = ok(
        {
            "data": [
                {"id": "claude-sonnet-4-6", "display_name": "Sonnet 4.6"},
                {"id": "claude-haiku-4-5"},
            ]
        }
    )
    result = _run(cfg, transport)
    assert [m.raw_id for m in result.models] == ["claude-sonnet-4-6", "claude-haiku-4-5"]
    assert result.models[0].litellm_id == "anthropic/claude-sonnet-4-6"


def test_anthropic_adapter_requires_api_key_env() -> None:
    cfg = ProviderConfig.model_validate({})
    with pytest.raises(DiscoveryError, match="api_key_env required"):
        _run(cfg, ok({"data": []}))


def test_anthropic_adapter_uses_x_api_key_for_regular_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Standard `sk-ant-api...` keys go on the official `x-api-key` header."""
    monkeypatch.setenv("ANTHROPIC_KEY", "sk-ant-api03-test")
    seen: dict[str, str] = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        seen.update(dict(request.headers))
        return httpx.Response(200, json={"data": []})

    cfg = ProviderConfig.model_validate({"api_key_env": "ANTHROPIC_KEY"})
    _run(cfg, httpx.MockTransport(_capture))
    assert seen.get("x-api-key") == "sk-ant-api03-test"
    assert "authorization" not in seen
    assert "anthropic-beta" not in seen


def test_anthropic_adapter_uses_bearer_for_oauth_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claude-Code OAuth tokens (`sk-ant-oat...`) require Bearer + beta header.

    api.anthropic.com 401s with `invalid x-api-key` for OAuth credentials
    sent on the `x-api-key` header; the only accepted shape is
    `Authorization: Bearer ...` plus `anthropic-beta: oauth-2025-04-20`.
    """
    monkeypatch.setenv("ANTHROPIC_KEY", "sk-ant-oat01-deadbeef")
    seen: dict[str, str] = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        seen.update(dict(request.headers))
        return httpx.Response(200, json={"data": []})

    cfg = ProviderConfig.model_validate({"api_key_env": "ANTHROPIC_KEY"})
    _run(cfg, httpx.MockTransport(_capture))
    assert seen.get("authorization") == "Bearer sk-ant-oat01-deadbeef"
    assert seen.get("anthropic-beta") == "oauth-2025-04-20"
    assert "x-api-key" not in seen


def test_anthropic_adapter_stamps_litellm_id_on_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provenance: discovery must be tagged in merge sources for anthropic too."""
    monkeypatch.setenv("ANTHROPIC_KEY", "sk-ant-test")
    cfg = ProviderConfig.model_validate({"api_key_env": "ANTHROPIC_KEY"})
    transport = ok({"data": [{"id": "claude-sonnet-4-6"}]})
    result = _run(cfg, transport)
    assert result.models[0].partial.litellm_id == "anthropic/claude-sonnet-4-6"
