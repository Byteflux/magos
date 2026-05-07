"""``magos.shapes`` package: spec lookup + endpoint -> shape mapping."""

from __future__ import annotations

import pytest

from magos.shapes import ANTHROPIC, OPENAI_CHAT, OPENAI_RESPONSES, SHAPES, shape_for_endpoint


def test_shapes_dict_keys_match_spec_names() -> None:
    assert set(SHAPES) == {"anthropic", "openai-chat", "openai-responses"}
    assert SHAPES["anthropic"] is ANTHROPIC
    assert SHAPES["openai-chat"] is OPENAI_CHAT
    assert SHAPES["openai-responses"] is OPENAI_RESPONSES


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("/v1/messages", "anthropic"),
        ("/v1/chat/completions", "openai-chat"),
        ("/v1/responses", "openai-responses"),
        ("/v1/responses/{id}", "openai-responses"),
        ("/v1/messages/count_tokens", None),
        ("/v1/responses/{id}/input_items", None),
    ],
)
def test_shape_for_endpoint(endpoint: str, expected: str | None) -> None:
    assert shape_for_endpoint(endpoint) == expected


def test_specs_are_frozen() -> None:
    """``ShapeSpec`` is a frozen dataclass so field lookups are stable."""
    with pytest.raises(AttributeError):
        ANTHROPIC.system_field = "other"  # type: ignore[misc]


def test_anthropic_has_cache_write_only_shape() -> None:
    """``cache_write`` is Anthropic-only; OpenAI shapes omit the key."""
    assert "cache_write" in ANTHROPIC.usage_keys
    assert "cache_write" not in OPENAI_CHAT.usage_keys
    assert "cache_write" not in OPENAI_RESPONSES.usage_keys


def test_compression_provider_grouping() -> None:
    """Both OpenAI shapes share the ``openai`` compression provider."""
    assert ANTHROPIC.compression_provider == "anthropic"
    assert OPENAI_CHAT.compression_provider == "openai"
    assert OPENAI_RESPONSES.compression_provider == "openai"
