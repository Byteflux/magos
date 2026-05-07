"""``OpenAIAdapter`` discovery tests."""

from __future__ import annotations

import httpx
import pytest

from magos.registry.discovery.base import DiscoveryError, DiscoveryResult
from magos.registry.discovery.openai import OpenAIAdapter
from magos.registry.schema import ProviderConfig
from tests.registry.discovery._helpers import err, ok, run_discover


def _run(cfg: ProviderConfig, transport: httpx.MockTransport) -> DiscoveryResult:
    return run_discover(OpenAIAdapter(), "openai", cfg, transport)


def test_openai_adapter_maps_data_ids_to_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OAI_KEY", "sk-test")
    cfg = ProviderConfig.model_validate(
        {"api_key_env": "OAI_KEY", "base_url": "http://localhost:8001"}
    )
    transport = ok({"data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}, {"id": ""}]})
    result = _run(cfg, transport)
    assert [m.raw_id for m in result.models] == ["gpt-4o", "gpt-4o-mini"]
    assert all(m.litellm_id.startswith("openai/") for m in result.models)


def test_openai_adapter_uses_litellm_provider_override() -> None:
    cfg = ProviderConfig.model_validate(
        {"base_url": "http://vllm:8000", "litellm_provider": "hosted_vllm"}
    )
    transport = ok({"data": [{"id": "llama-3-70b"}]})
    result = _run(cfg, transport)
    assert result.models[0].litellm_id == "hosted_vllm/llama-3-70b"


def test_openai_adapter_defaults_to_api_openai_com_when_base_url_unset() -> None:
    """Operators omitting ``base_url`` hit api.openai.com.

    The overwhelmingly common case for ``discovery: openai`` is OpenAI
    proper; self-hosted OpenAI-shape backends explicitly set their own
    base_url because they're not on api.openai.com. Defaulting here
    removes a yaml line for the common case without surprising the
    self-hosted users.
    """
    seen: dict[str, str] = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"data": []})

    cfg = ProviderConfig.model_validate({})
    _run(cfg, httpx.MockTransport(_capture))
    assert seen["url"] == "https://api.openai.com/v1/models"


def test_openai_adapter_raises_on_http_error() -> None:
    cfg = ProviderConfig.model_validate({"base_url": "http://localhost:8001"})
    with pytest.raises(DiscoveryError, match="HTTP 401"):
        _run(cfg, err(401, "unauthorized"))


def test_openai_adapter_raises_when_data_missing() -> None:
    cfg = ProviderConfig.model_validate({"base_url": "http://localhost:8001"})
    with pytest.raises(DiscoveryError, match="missing or non-list 'data'"):
        _run(cfg, ok({"object": "list"}))


def test_openai_adapter_raises_when_env_var_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_KEY", raising=False)
    cfg = ProviderConfig.model_validate(
        {"api_key_env": "MISSING_KEY", "base_url": "http://localhost:8001"}
    )
    with pytest.raises(DiscoveryError, match="env var MISSING_KEY unset"):
        _run(cfg, ok({"data": []}))


def test_openai_adapter_stamps_litellm_id_on_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provenance: discovery must be tagged in merge sources for openai too."""
    monkeypatch.setenv("OAI_KEY", "sk-test")
    cfg = ProviderConfig.model_validate(
        {"api_key_env": "OAI_KEY", "base_url": "http://localhost:8001"}
    )
    transport = ok({"data": [{"id": "gpt-4o"}]})
    result = _run(cfg, transport)
    assert result.models[0].partial.litellm_id == "openai/gpt-4o"
