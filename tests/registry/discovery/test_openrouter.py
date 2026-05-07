"""``OpenRouterAdapter`` discovery tests."""

from __future__ import annotations

import httpx
import pytest

from magos.registry.discovery.base import DiscoveryResult
from magos.registry.discovery.openrouter import OpenRouterAdapter
from magos.registry.schema import ProviderConfig

from ._helpers import ok, run_discover


def _run(cfg: ProviderConfig, transport: httpx.MockTransport) -> DiscoveryResult:
    return run_discover(OpenRouterAdapter(), "openrouter", cfg, transport)


def test_openrouter_adapter_populates_partial_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OR_KEY", "sk-or-test")
    cfg = ProviderConfig.model_validate({"api_key_env": "OR_KEY"})
    transport = ok(
        {
            "data": [
                {
                    "id": "anthropic/claude-sonnet-4-6",
                    "context_length": 200000,
                    "pricing": {
                        "prompt": "0.000003",
                        "completion": "0.000015",
                        "input_cache_read": "0.0000003",
                        "input_cache_write": "0.00000375",
                    },
                    "architecture": {"modality": "text+image->text"},
                    "top_provider": {"max_completion_tokens": 8192},
                }
            ]
        }
    )
    result = _run(cfg, transport)
    assert len(result.models) == 1
    m = result.models[0]
    assert m.raw_id == "anthropic/claude-sonnet-4-6"
    assert m.litellm_id == "openrouter/anthropic/claude-sonnet-4-6"
    assert m.partial.context_size == 200000
    assert m.partial.max_output == 8192
    # 0.000003 USD per token -> 3.0 USD per million tokens.
    assert m.partial.input_cost == pytest.approx(3.0)
    assert m.partial.output_cost == pytest.approx(15.0)
    assert m.partial.cache_read_cost == pytest.approx(0.30)
    assert m.partial.cache_write_cost == pytest.approx(3.75)
    assert m.partial.input_modalities == ("text", "image")
    assert m.partial.output_modalities == ("text",)


def test_openrouter_adapter_prefers_explicit_modality_arrays() -> None:
    """The new ``architecture.{input,output}_modalities`` arrays win over the legacy string."""
    cfg = ProviderConfig.model_validate({})
    transport = ok(
        {
            "data": [
                {
                    "id": "x/y",
                    "architecture": {
                        # Legacy field: input=text, output=image (would lose info).
                        "modality": "text->image",
                        # New explicit fields override the split.
                        "input_modalities": ["text", "image"],
                        "output_modalities": ["text", "image"],
                    },
                }
            ]
        }
    )
    result = _run(cfg, transport)
    m = result.models[0]
    assert m.partial.input_modalities == ("text", "image")
    assert m.partial.output_modalities == ("text", "image")


def test_openrouter_adapter_falls_back_to_legacy_modality_string() -> None:
    cfg = ProviderConfig.model_validate({})
    transport = ok(
        {
            "data": [
                {
                    "id": "x/y",
                    "architecture": {"modality": "text+image->text"},
                }
            ]
        }
    )
    result = _run(cfg, transport)
    m = result.models[0]
    assert m.partial.input_modalities == ("text", "image")
    assert m.partial.output_modalities == ("text",)


def test_openrouter_adapter_omits_cache_costs_when_pricing_block_lacks_them() -> None:
    cfg = ProviderConfig.model_validate({})
    transport = ok(
        {
            "data": [
                {
                    "id": "x/y",
                    "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                }
            ]
        }
    )
    result = _run(cfg, transport)
    m = result.models[0]
    assert m.partial.cache_read_cost is None
    assert m.partial.cache_write_cost is None


def test_openrouter_adapter_handles_missing_optional_fields() -> None:
    cfg = ProviderConfig.model_validate({})
    transport = ok({"data": [{"id": "x/y"}]})
    result = _run(cfg, transport)
    assert result.models[0].partial.context_size is None


def test_openrouter_adapter_drops_negative_pricing_sentinels() -> None:
    """OpenRouter uses -1 for meta routes (auto, etc.); must not leak as cost."""
    cfg = ProviderConfig.model_validate({})
    transport = ok(
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
    result = _run(cfg, transport)
    m = result.models[0]
    assert m.partial.input_cost is None
    assert m.partial.output_cost is None


def test_openrouter_adapter_drops_max_output_when_exceeds_context() -> None:
    """Self-inconsistent payloads (output > total context) drop max_output."""
    cfg = ProviderConfig.model_validate({})
    transport = ok(
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
    result = _run(cfg, transport)
    m = result.models[0]
    assert m.partial.context_size == 131072
    assert m.partial.max_output is None
