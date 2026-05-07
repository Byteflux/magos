"""``VultrAdapter`` discovery tests."""

from __future__ import annotations

import httpx
import pytest

from magos.registry.discovery.base import DiscoveryError, DiscoveryResult
from magos.registry.discovery.vultr import VultrAdapter
from magos.registry.schema import ProviderConfig
from tests.registry.discovery._helpers import ok, run_discover


def _run(cfg: ProviderConfig, transport: httpx.MockTransport) -> DiscoveryResult:
    return run_discover(VultrAdapter(), "vultr", cfg, transport)


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
    result = _run(cfg, httpx.MockTransport(_h))
    assert captured["url"].endswith("/v1/models/lookup")
    m = result.models[0]
    assert m.raw_id == "MiniMaxAI/MiniMax-M2.7"
    assert m.litellm_id == "custom_openai/MiniMaxAI/MiniMax-M2.7"
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
    _run(cfg, httpx.MockTransport(_h))
    assert captured["url"] == "https://api.vultrinference.com/v1/models/lookup"


def test_vultr_adapter_raises_when_models_field_missing() -> None:
    cfg = ProviderConfig.model_validate({})
    with pytest.raises(DiscoveryError, match="missing or non-list 'models'"):
        _run(cfg, ok({"object": "list"}))


def test_vultr_adapter_drops_negative_cost() -> None:
    cfg = ProviderConfig.model_validate({})
    transport = ok(
        {"models": [{"id": "x/y", "context_length": 4096, "cost_input": -1, "cost_output": -1}]}
    )
    result = _run(cfg, transport)
    m = result.models[0]
    assert m.partial.input_cost is None
    assert m.partial.output_cost is None
