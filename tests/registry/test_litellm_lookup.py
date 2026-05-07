"""Tests for `magos.registry.litellm_lookup` with injected fakes."""

from __future__ import annotations

from typing import Any

from magos.registry.litellm_lookup import PartialEntry, lookup


def _info(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "max_input_tokens": 200000,
        "max_output_tokens": 8192,
        "input_cost_per_token": 3e-6,
        "output_cost_per_token": 1.5e-5,
    }
    base.update(overrides)
    return base


def test_lookup_populates_all_known_fields() -> None:
    def fake(model: str) -> dict[str, Any]:
        return _info(
            supports_vision=True,
            supports_audio_input=False,
            cache_read_input_token_cost=3e-7,
            cache_creation_input_token_cost=3.75e-6,
        )

    result = lookup("anthropic/claude-sonnet-4-6", get_info=fake)
    assert result == PartialEntry(
        litellm_id="anthropic/claude-sonnet-4-6",
        context_size=200000,
        max_output=8192,
        input_cost=3.0,
        output_cost=15.0,
        cache_read_cost=0.3,
        cache_write_cost=3.75,
        input_modalities=("text", "image"),
        output_modalities=("text",),
    )


def test_lookup_leaves_cache_costs_none_when_litellm_omits_them() -> None:
    def fake(model: str) -> dict[str, Any]:
        # No cache_*_token_cost keys: common for non-Anthropic providers.
        return _info()

    result = lookup("openai/gpt-4o", get_info=fake)
    assert result.cache_read_cost is None
    assert result.cache_write_cost is None


def test_lookup_falls_back_to_max_tokens_when_max_input_missing() -> None:
    def fake(model: str) -> dict[str, Any]:
        info = _info()
        del info["max_input_tokens"]
        info["max_tokens"] = 100000
        return info

    result = lookup("openai/gpt-4o", get_info=fake)
    assert result.context_size == 100000


def test_lookup_returns_empty_partial_on_value_error() -> None:
    def fake(model: str) -> dict[str, Any]:
        raise ValueError("model not found")

    result = lookup("unknown/model", get_info=fake)
    assert result == PartialEntry()


def test_lookup_returns_empty_partial_on_unexpected_exception() -> None:
    def fake(model: str) -> dict[str, Any]:
        raise RuntimeError("boom")

    result = lookup("unknown/model", get_info=fake)
    assert result == PartialEntry()


def test_lookup_includes_text_modality_by_default() -> None:
    def fake(model: str) -> dict[str, Any]:
        return _info()

    result = lookup("anthropic/claude-haiku-4-5", get_info=fake)
    assert result.input_modalities == ("text",)
    assert result.output_modalities == ("text",)


def test_lookup_promotes_audio_output_when_litellm_flags_it() -> None:
    def fake(model: str) -> dict[str, Any]:
        return _info(supports_audio_output=True)

    result = lookup("openai/gpt-4o-audio", get_info=fake)
    assert result.input_modalities == ("text",)
    assert result.output_modalities == ("text", "audio")
