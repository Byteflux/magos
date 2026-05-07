"""Dispatcher routing: Anthropic upstream goes through ``litellm.anthropic_messages``,
everything else through ``acompletion`` + adapter translation.
"""

from __future__ import annotations

import asyncio
from typing import Any

import litellm
import pytest
from litellm.types.utils import Choices, Message, ModelResponse, Usage

from magos.dispatch.translate.anthropic import _dispatch_anthropic_messages


@pytest.mark.unit
def test_dispatch_routes_non_anthropic_via_acompletion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-Anthropic dispatch must go through ``litellm.acompletion``,
    not ``litellm.anthropic_messages``.

    Background: LiteLLM's ``anthropic_messages`` adapter chain leaks
    the LiteLLM provider prefix into the outbound body for non-
    Anthropic upstreams (sends ``model: 'openrouter/qwen/...'`` where
    OpenRouter expects ``model: 'qwen/...'``). ``acompletion`` strips
    the prefix correctly. The dispatcher must therefore route non-
    Anthropic traffic via ``acompletion`` + manual translation.
    """
    seen_anthropic_messages: list[dict[str, Any]] = []
    seen_acompletion: list[dict[str, Any]] = []

    async def fake_anthropic_messages(**kwargs: Any) -> Any:
        seen_anthropic_messages.append(kwargs)
        return {"id": "anthropic", "content": []}

    async def fake_acompletion(**kwargs: Any) -> Any:
        seen_acompletion.append(kwargs)
        return ModelResponse(
            id="x",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(role="assistant", content="ok"),
                )
            ],
            model="qwen/qwen3-coder",
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    monkeypatch.setattr(litellm, "anthropic_messages", fake_anthropic_messages)
    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    # Anthropic dispatch -> uses anthropic_messages
    asyncio.run(
        _dispatch_anthropic_messages(
            model="anthropic/claude-opus-4-7",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
        )
    )
    assert len(seen_anthropic_messages) == 1
    assert len(seen_acompletion) == 0

    # Non-Anthropic (OpenRouter) -> uses acompletion
    asyncio.run(
        _dispatch_anthropic_messages(
            model="openrouter/qwen/qwen3-coder",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
        )
    )
    assert len(seen_anthropic_messages) == 1  # unchanged
    assert len(seen_acompletion) == 1
    # The model passed to acompletion still has the prefix; acompletion
    # itself strips it before the outbound HTTP call. The bug we're
    # working around is specific to anthropic_messages, not acompletion.
    assert seen_acompletion[0]["model"] == "openrouter/qwen/qwen3-coder"
