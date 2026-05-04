"""Unit tests for ``magos.egress.tokens.count_tokens``.

Verifies the seam contract: payload composition (model rewrite, optional
``system``/``tools``/``tool_choice`` forwarding) and result coercion
(handles both pydantic-like ``TokenCountResponse`` objects and plain dicts).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from magos.egress import tokens

SIMPLE_REQUEST: dict[str, Any] = {
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "Hello, world."}],
}


@pytest.mark.unit
def test_count_tokens_threads_dispatch_model_and_messages() -> None:
    received: dict[str, Any] = {}

    async def fake(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return {"total_tokens": 7}

    n = asyncio.run(
        tokens.count_tokens(
            SIMPLE_REQUEST,
            dispatch_model="anthropic/claude-3-5-sonnet-20241022",
            count=fake,
        )
    )
    assert n == 7
    assert received["model"] == "anthropic/claude-3-5-sonnet-20241022"
    assert received["messages"] == SIMPLE_REQUEST["messages"]


@pytest.mark.unit
def test_count_tokens_forwards_optional_fields_when_present() -> None:
    received: dict[str, Any] = {}

    async def fake(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return {"total_tokens": 1}

    body = {
        **SIMPLE_REQUEST,
        "system": "You are concise.",
        "tools": [{"name": "x", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "any"},
    }
    asyncio.run(tokens.count_tokens(body, dispatch_model="anthropic/foo", count=fake))
    assert received["system"] == "You are concise."
    assert received["tools"][0]["name"] == "x"
    assert received["tool_choice"] == {"type": "any"}


@pytest.mark.unit
def test_count_tokens_omits_optional_fields_when_absent() -> None:
    received: dict[str, Any] = {}

    async def fake(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return {"total_tokens": 1}

    asyncio.run(tokens.count_tokens(SIMPLE_REQUEST, dispatch_model="anthropic/foo", count=fake))
    for optional in ("system", "tools", "tool_choice"):
        assert optional not in received


@pytest.mark.unit
def test_count_tokens_coerces_pydantic_like_response() -> None:
    class _Resp:
        total_tokens = 13

    async def fake(**_: Any) -> Any:
        return _Resp()

    n = asyncio.run(tokens.count_tokens(SIMPLE_REQUEST, dispatch_model="anthropic/foo", count=fake))
    assert n == 13


@pytest.mark.unit
def test_count_tokens_rejects_unsupported_response_type() -> None:
    async def fake(**_: Any) -> Any:
        return "not a dict"

    with pytest.raises(TypeError, match="unsupported type"):
        asyncio.run(tokens.count_tokens(SIMPLE_REQUEST, dispatch_model="anthropic/foo", count=fake))
