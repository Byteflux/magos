"""`magos.shapes` package: Shape lookup + endpoint -> shape mapping."""

from __future__ import annotations

import pytest

from magos.shapes import (
    ANTHROPIC,
    OPENAI_CHAT,
    OPENAI_RESPONSES,
    SHAPES,
    Shape,
    shape_by_name,
    shape_for_endpoint,
)


def test_shapes_tuple_contains_all_three() -> None:
    assert ANTHROPIC in SHAPES
    assert OPENAI_CHAT in SHAPES
    assert OPENAI_RESPONSES in SHAPES
    assert len(SHAPES) == 3


def test_shape_by_name_returns_correct_instances() -> None:
    assert shape_by_name("anthropic") is ANTHROPIC
    assert shape_by_name("openai-chat") is OPENAI_CHAT
    assert shape_by_name("openai-responses") is OPENAI_RESPONSES


def test_shape_by_name_returns_none_for_unknown() -> None:
    assert shape_by_name("unknown") is None
    assert shape_by_name("") is None
    assert shape_by_name("openai") is None


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("/v1/messages", ANTHROPIC),
        ("/v1/chat/completions", OPENAI_CHAT),
        ("/v1/responses", OPENAI_RESPONSES),
        ("/v1/responses/{id}", OPENAI_RESPONSES),
        ("/v1/messages/count_tokens", None),
        ("/v1/responses/{id}/input_items", None),
    ],
)
def test_shape_for_endpoint(endpoint: str, expected: Shape | None) -> None:
    assert shape_for_endpoint(endpoint) is expected


def test_specs_are_frozen() -> None:
    """`Shape` is a frozen dataclass so field lookups are stable."""
    with pytest.raises(AttributeError):
        ANTHROPIC.system_field = "other"  # type: ignore[misc]


def test_anthropic_has_cache_write_only_shape() -> None:
    """`cache_write` is Anthropic-only; OpenAI shapes omit the key."""
    assert "cache_write" in ANTHROPIC.usage_keys
    assert "cache_write" not in OPENAI_CHAT.usage_keys
    assert "cache_write" not in OPENAI_RESPONSES.usage_keys


def test_compression_provider_grouping() -> None:
    """Both OpenAI shapes share the `openai` compression provider."""
    assert ANTHROPIC.compression_provider == "anthropic"
    assert OPENAI_CHAT.compression_provider == "openai"
    assert OPENAI_RESPONSES.compression_provider == "openai"
