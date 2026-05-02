"""Tests for individual discovery adapters using httpx.MockTransport."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from magos.registry.discovery.anthropic import AnthropicAdapter
from magos.registry.discovery.base import DiscoveryError, DiscoveryResult
from magos.registry.discovery.factory import adapter_for
from magos.registry.discovery.noop import NoopAdapter
from magos.registry.discovery.openai import OpenAIAdapter
from magos.registry.discovery.openrouter import OpenRouterAdapter
from magos.registry.discovery.vultr import VultrAdapter
from magos.registry.schema import ProviderConfig


def _ok(payload: dict[str, Any]) -> httpx.MockTransport:
    def _h(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(_h)


def _err(status: int, body: str = "") -> httpx.MockTransport:
    def _h(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=body)

    return httpx.MockTransport(_h)


async def _run_openai(cfg: ProviderConfig, transport: httpx.MockTransport) -> DiscoveryResult:
    async with httpx.AsyncClient(transport=transport) as client:
        return await OpenAIAdapter().discover("openai", cfg, client)


async def _run_anthropic(cfg: ProviderConfig, transport: httpx.MockTransport) -> DiscoveryResult:
    async with httpx.AsyncClient(transport=transport) as client:
        return await AnthropicAdapter().discover("anthropic", cfg, client)


async def _run_openrouter(cfg: ProviderConfig, transport: httpx.MockTransport) -> DiscoveryResult:
    async with httpx.AsyncClient(transport=transport) as client:
        return await OpenRouterAdapter().discover("openrouter", cfg, client)


async def _run_vultr(cfg: ProviderConfig, transport: httpx.MockTransport) -> DiscoveryResult:
    async with httpx.AsyncClient(transport=transport) as client:
        return await VultrAdapter().discover("vultr", cfg, client)


def test_openai_adapter_maps_data_ids_to_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OAI_KEY", "sk-test")
    cfg = ProviderConfig.model_validate(
        {"api_key_env": "OAI_KEY", "base_url": "http://localhost:8001"}
    )
    transport = _ok({"data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}, {"id": ""}]})
    result = asyncio.run(_run_openai(cfg, transport))
    assert [m.raw_id for m in result.models] == ["gpt-4o", "gpt-4o-mini"]
    assert all(m.litellm_id.startswith("openai/") for m in result.models)


def test_openai_adapter_uses_litellm_provider_override() -> None:
    cfg = ProviderConfig.model_validate(
        {"base_url": "http://vllm:8000", "litellm_provider": "hosted_vllm"}
    )
    transport = _ok({"data": [{"id": "llama-3-70b"}]})
    result = asyncio.run(_run_openai(cfg, transport))
    assert result.models[0].litellm_id == "hosted_vllm/llama-3-70b"


def test_openai_adapter_requires_base_url() -> None:
    cfg = ProviderConfig.model_validate({})
    with pytest.raises(DiscoveryError, match="base_url required"):
        asyncio.run(_run_openai(cfg, _ok({"data": []})))


def test_openai_adapter_raises_on_http_error() -> None:
    cfg = ProviderConfig.model_validate({"base_url": "http://localhost:8001"})
    with pytest.raises(DiscoveryError, match="HTTP 401"):
        asyncio.run(_run_openai(cfg, _err(401, "unauthorized")))


def test_openai_adapter_raises_when_data_missing() -> None:
    cfg = ProviderConfig.model_validate({"base_url": "http://localhost:8001"})
    with pytest.raises(DiscoveryError, match="missing or non-list 'data'"):
        asyncio.run(_run_openai(cfg, _ok({"object": "list"})))


def test_openai_adapter_raises_when_env_var_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_KEY", raising=False)
    cfg = ProviderConfig.model_validate(
        {"api_key_env": "MISSING_KEY", "base_url": "http://localhost:8001"}
    )
    with pytest.raises(DiscoveryError, match="env var MISSING_KEY unset"):
        asyncio.run(_run_openai(cfg, _ok({"data": []})))


def test_anthropic_adapter_extracts_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_KEY", "sk-ant-test")
    cfg = ProviderConfig.model_validate({"api_key_env": "ANTHROPIC_KEY"})
    transport = _ok(
        {
            "data": [
                {"id": "claude-sonnet-4-6", "display_name": "Sonnet 4.6"},
                {"id": "claude-haiku-4-5"},
            ]
        }
    )
    result = asyncio.run(_run_anthropic(cfg, transport))
    assert [m.raw_id for m in result.models] == ["claude-sonnet-4-6", "claude-haiku-4-5"]
    assert result.models[0].litellm_id == "anthropic/claude-sonnet-4-6"


def test_anthropic_adapter_requires_api_key_env() -> None:
    cfg = ProviderConfig.model_validate({})
    with pytest.raises(DiscoveryError, match="api_key_env required"):
        asyncio.run(_run_anthropic(cfg, _ok({"data": []})))


def test_openrouter_adapter_populates_partial_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OR_KEY", "sk-or-test")
    cfg = ProviderConfig.model_validate({"api_key_env": "OR_KEY"})
    transport = _ok(
        {
            "data": [
                {
                    "id": "anthropic/claude-sonnet-4-6",
                    "context_length": 200000,
                    "pricing": {"prompt": "0.000003", "completion": "0.000015"},
                    "architecture": {"modality": "text+image->text"},
                    "top_provider": {"max_completion_tokens": 8192},
                }
            ]
        }
    )
    result = asyncio.run(_run_openrouter(cfg, transport))
    assert len(result.models) == 1
    m = result.models[0]
    assert m.raw_id == "anthropic/claude-sonnet-4-6"
    assert m.litellm_id == "openrouter/anthropic/claude-sonnet-4-6"
    assert m.partial.context_size == 200000
    assert m.partial.max_output == 8192
    # 0.000003 USD per token -> 3.0 USD per million tokens.
    assert m.partial.input_cost == pytest.approx(3.0)
    assert m.partial.output_cost == pytest.approx(15.0)
    assert m.partial.modalities == ("text", "image")


def test_openrouter_adapter_handles_missing_optional_fields() -> None:
    cfg = ProviderConfig.model_validate({})
    transport = _ok({"data": [{"id": "x/y"}]})
    result = asyncio.run(_run_openrouter(cfg, transport))
    assert result.models[0].partial.context_size is None


def test_openai_adapter_stamps_litellm_id_on_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provenance: discovery must be tagged in merge sources for openai too."""
    monkeypatch.setenv("OAI_KEY", "sk-test")
    cfg = ProviderConfig.model_validate(
        {"api_key_env": "OAI_KEY", "base_url": "http://localhost:8001"}
    )
    transport = _ok({"data": [{"id": "gpt-4o"}]})
    result = asyncio.run(_run_openai(cfg, transport))
    assert result.models[0].partial.litellm_id == "openai/gpt-4o"


def test_anthropic_adapter_stamps_litellm_id_on_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provenance: discovery must be tagged in merge sources for anthropic too."""
    monkeypatch.setenv("ANTHROPIC_KEY", "sk-ant-test")
    cfg = ProviderConfig.model_validate({"api_key_env": "ANTHROPIC_KEY"})
    transport = _ok({"data": [{"id": "claude-sonnet-4-6"}]})
    result = asyncio.run(_run_anthropic(cfg, transport))
    assert result.models[0].partial.litellm_id == "anthropic/claude-sonnet-4-6"


def test_openrouter_adapter_drops_negative_pricing_sentinels() -> None:
    """OpenRouter uses -1 for meta routes (auto, etc.); must not leak as cost."""
    cfg = ProviderConfig.model_validate({})
    transport = _ok(
        {
            "data": [
                {
                    "id": "openrouter/auto",
                    "context_length": 200000,
                    "pricing": {"prompt": "-1", "completion": "-1"},
                }
            ]
        }
    )
    result = asyncio.run(_run_openrouter(cfg, transport))
    m = result.models[0]
    assert m.partial.input_cost is None
    assert m.partial.output_cost is None


def test_openrouter_adapter_drops_max_output_when_exceeds_context() -> None:
    """Self-inconsistent payloads (output > total context) drop max_output."""
    cfg = ProviderConfig.model_validate({})
    transport = _ok(
        {
            "data": [
                {
                    "id": "weird/model",
                    "context_length": 131072,
                    "top_provider": {"max_completion_tokens": 262144},
                }
            ]
        }
    )
    result = asyncio.run(_run_openrouter(cfg, transport))
    m = result.models[0]
    assert m.partial.context_size == 131072
    assert m.partial.max_output is None


def test_vultr_adapter_populates_partial_from_lookup_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lookup endpoint provides context_length plus cents-per-million pricing."""
    monkeypatch.setenv("VULTR_KEY", "sk-vultr-test")

    captured: dict[str, str] = {}

    def _h(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "id": "MiniMaxAI/MiniMax-M2.7",
                        "context_length": 1048576,
                        "cost_input": 30,
                        "cost_output": 120,
                    }
                ]
            },
        )

    cfg = ProviderConfig.model_validate({"api_key_env": "VULTR_KEY"})
    result = asyncio.run(_run_vultr(cfg, httpx.MockTransport(_h)))
    assert captured["url"].endswith("/v1/models/lookup")
    m = result.models[0]
    assert m.raw_id == "MiniMaxAI/MiniMax-M2.7"
    assert m.litellm_id == "openai/MiniMaxAI/MiniMax-M2.7"
    assert m.partial.context_size == 1048576
    # 30 cents per million tokens -> $0.30 per million tokens.
    assert m.partial.input_cost == pytest.approx(0.30)
    assert m.partial.output_cost == pytest.approx(1.20)


def test_vultr_adapter_handles_v1_suffix_in_base_url() -> None:
    """base_url ending in /v1 should not produce //v1/v1/... double-prefix."""
    captured: dict[str, str] = {}

    def _h(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"models": []})

    cfg = ProviderConfig.model_validate({"base_url": "https://api.vultrinference.com/v1"})
    asyncio.run(_run_vultr(cfg, httpx.MockTransport(_h)))
    assert captured["url"] == "https://api.vultrinference.com/v1/models/lookup"


def test_vultr_adapter_raises_when_models_field_missing() -> None:
    cfg = ProviderConfig.model_validate({})
    with pytest.raises(DiscoveryError, match="missing or non-list 'models'"):
        asyncio.run(_run_vultr(cfg, _ok({"object": "list"})))


def test_vultr_adapter_drops_negative_cost() -> None:
    cfg = ProviderConfig.model_validate({})
    transport = _ok(
        {"models": [{"id": "x/y", "context_length": 4096, "cost_input": -1, "cost_output": -1}]}
    )
    result = asyncio.run(_run_vultr(cfg, transport))
    m = result.models[0]
    assert m.partial.input_cost is None
    assert m.partial.output_cost is None


def test_noop_adapter_returns_empty_result() -> None:
    cfg = ProviderConfig.model_validate({})

    async def _run() -> None:
        async with httpx.AsyncClient(transport=_err(500)) as client:
            result = await NoopAdapter().discover("manual", cfg, client)
            assert result.models == ()

    asyncio.run(_run())


def test_adapter_for_resolves_each_known_kind() -> None:
    assert isinstance(
        adapter_for(ProviderConfig.model_validate({"discovery": "openai"})),
        OpenAIAdapter,
    )
    assert isinstance(
        adapter_for(ProviderConfig.model_validate({"discovery": "anthropic"})),
        AnthropicAdapter,
    )
    assert isinstance(
        adapter_for(ProviderConfig.model_validate({"discovery": "openrouter"})),
        OpenRouterAdapter,
    )
    assert isinstance(
        adapter_for(ProviderConfig.model_validate({"discovery": "vultr"})),
        VultrAdapter,
    )
    assert isinstance(adapter_for(ProviderConfig.model_validate({})), NoopAdapter)
