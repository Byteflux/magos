"""Tests for magos.tokens.

Exercises the local estimator against a small fixture, the passthrough
dispatch (mocked Anthropic SDK), provider gating, and the fallback behaviour
when passthrough raises.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from magos import tokens

SIMPLE_REQUEST: dict[str, Any] = {
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 16,
    "messages": [{"role": "user", "content": "Hello, world."}],
}

OPENAI_MODEL_REQUEST: dict[str, Any] = {
    **SIMPLE_REQUEST,
    "model": "gpt-4o-mini",
}


@pytest.mark.unit
def test_count_locally_returns_positive_int() -> None:
    n = tokens.count_locally(SIMPLE_REQUEST)
    assert isinstance(n, int)
    assert n > 0


@pytest.mark.unit
def test_count_locally_works_without_max_tokens() -> None:
    """count_tokens body has no ``max_tokens`` field per Anthropic's spec."""
    body = {k: v for k, v in SIMPLE_REQUEST.items() if k != "max_tokens"}
    assert "max_tokens" not in body
    n = tokens.count_locally(body)
    assert n == tokens.count_locally(SIMPLE_REQUEST)


@pytest.mark.unit
def test_count_input_tokens_local_when_passthrough_disabled() -> None:
    n = asyncio.run(tokens.count_input_tokens(SIMPLE_REQUEST))
    assert n == tokens.count_locally(SIMPLE_REQUEST)


@pytest.mark.unit
def test_count_input_tokens_passthrough_for_allowed_provider() -> None:
    captured: dict[str, Any] = {}

    async def fake_passthrough(
        req: dict[str, Any], *, forward_headers: dict[str, str] | None = None
    ) -> int:
        captured["called"] = True
        captured["model"] = req["model"]
        captured["forward_headers"] = forward_headers
        return 999

    with patch.dict(tokens.PASSTHROUGH_DISPATCH, {"anthropic": fake_passthrough}):
        n = asyncio.run(
            tokens.count_input_tokens(
                SIMPLE_REQUEST, passthrough_providers=frozenset({"anthropic"})
            )
        )

    assert n == 999
    assert captured["called"] is True
    assert captured["model"] == "claude-3-5-sonnet-20241022"
    assert captured["forward_headers"] is None


@pytest.mark.unit
def test_count_input_tokens_forwards_headers() -> None:
    captured: dict[str, Any] = {}

    async def fake_passthrough(
        req: dict[str, Any], *, forward_headers: dict[str, str] | None = None
    ) -> int:
        captured["forward_headers"] = forward_headers
        return 7

    with patch.dict(tokens.PASSTHROUGH_DISPATCH, {"anthropic": fake_passthrough}):
        asyncio.run(
            tokens.count_input_tokens(
                SIMPLE_REQUEST,
                passthrough_providers=frozenset({"anthropic"}),
                forward_headers={"authorization": "Bearer abc", "anthropic-beta": "x,y"},
            )
        )

    assert captured["forward_headers"] == {
        "authorization": "Bearer abc",
        "anthropic-beta": "x,y",
    }


@pytest.mark.unit
def test_count_input_tokens_falls_back_to_local_when_provider_not_allowed() -> None:
    """Anthropic in dispatch but not in allow-list -> local path."""

    async def should_not_be_called(
        _: dict[str, Any], *, forward_headers: dict[str, str] | None = None
    ) -> int:
        raise AssertionError("passthrough should not have been invoked")

    with patch.dict(tokens.PASSTHROUGH_DISPATCH, {"anthropic": should_not_be_called}):
        n = asyncio.run(
            tokens.count_input_tokens(SIMPLE_REQUEST, passthrough_providers=frozenset({"openai"}))
        )

    assert n == tokens.count_locally(SIMPLE_REQUEST)


@pytest.mark.unit
def test_count_input_tokens_falls_back_to_local_for_unsupported_provider() -> None:
    """OpenAI model with anthropic-only dispatch -> local path."""
    n = asyncio.run(
        tokens.count_input_tokens(
            OPENAI_MODEL_REQUEST,
            passthrough_providers=frozenset({"anthropic"}),
        )
    )
    assert n == tokens.count_locally(OPENAI_MODEL_REQUEST)


@pytest.mark.unit
def test_count_input_tokens_falls_back_to_local_on_passthrough_error() -> None:
    async def boom(_: dict[str, Any], *, forward_headers: dict[str, str] | None = None) -> int:
        raise RuntimeError("network down")

    with patch.dict(tokens.PASSTHROUGH_DISPATCH, {"anthropic": boom}):
        n = asyncio.run(
            tokens.count_input_tokens(
                SIMPLE_REQUEST, passthrough_providers=frozenset({"anthropic"})
            )
        )

    assert n == tokens.count_locally(SIMPLE_REQUEST)
