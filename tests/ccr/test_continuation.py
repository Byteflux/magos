"""``make_continuation_callable`` builds an api_call_fn closure for headroom."""

from __future__ import annotations

import asyncio
from typing import Any

from magos.ccr import make_continuation_callable


def test_continuation_substitutes_messages_and_tools() -> None:
    """The returned closure substitutes the supplied messages/tools into a copy
    of the original body, then invokes proxy_translate with the same kwargs."""
    from magos.egress.translate import TRANSLATE_HANDLERS  # noqa: PLC0415

    captured: dict[str, Any] = {}

    async def fake_completion(**kwargs: Any) -> dict[str, Any]:
        captured["messages"] = kwargs.get("messages")
        captured["tools"] = kwargs.get("tools")
        return {
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

    original_body = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "hi"}],
    }
    adapter = TRANSLATE_HANDLERS["/v1/messages"]

    fn = make_continuation_callable(
        adapter=adapter,
        original_body=original_body,
        completion=fake_completion,
        dispatch_model="claude-sonnet-4-5",
        provider="anthropic",
        forward_headers={},
        api_key=None,
        api_base=None,
    )

    new_messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "thinking"},
        {"role": "user", "content": "now retrieve"},
    ]
    new_tools = [{"name": "headroom_retrieve"}]

    result = asyncio.run(fn(new_messages, new_tools))

    assert result["model"] == "anthropic/claude-sonnet-4-5"
    # The closure passed the substituted messages downstream.
    assert captured["messages"] == new_messages
    assert captured["tools"] == new_tools


def test_continuation_does_not_mutate_original_body() -> None:
    from magos.egress.translate import TRANSLATE_HANDLERS  # noqa: PLC0415

    async def fake_completion(**_: Any) -> dict[str, Any]:
        return {"model": "x", "content": [], "usage": {"input_tokens": 1, "output_tokens": 1}}

    original_body = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "old_tool"}],
    }
    adapter = TRANSLATE_HANDLERS["/v1/messages"]

    fn = make_continuation_callable(
        adapter=adapter,
        original_body=original_body,
        completion=fake_completion,
        dispatch_model="claude-sonnet-4-5",
        provider="anthropic",
        forward_headers={},
        api_key=None,
        api_base=None,
    )
    asyncio.run(fn([{"role": "user", "content": "new"}], [{"name": "headroom_retrieve"}]))

    # Original body untouched.
    assert original_body["messages"] == [{"role": "user", "content": "hi"}]
    assert original_body["tools"] == [{"name": "old_tool"}]


def test_continuation_drops_tools_when_none() -> None:
    """If headroom passes ``tools=None``, the closure removes the tools key
    from the substituted body (rather than passing tools=None through, which
    some adapters might object to)."""
    from magos.egress.translate import TRANSLATE_HANDLERS  # noqa: PLC0415

    captured: dict[str, Any] = {}

    async def fake_completion(**kwargs: Any) -> dict[str, Any]:
        captured["body_tools"] = kwargs.get("tools")
        return {"model": "x", "content": [], "usage": {"input_tokens": 1, "output_tokens": 1}}

    original_body = {
        "model": "x",
        "messages": [],
        "tools": [{"name": "old_tool"}],
    }
    adapter = TRANSLATE_HANDLERS["/v1/messages"]

    fn = make_continuation_callable(
        adapter=adapter,
        original_body=original_body,
        completion=fake_completion,
        dispatch_model="x",
        provider="anthropic",
        forward_headers={},
        api_key=None,
        api_base=None,
    )
    # Call must not crash with tools=None.
    asyncio.run(fn([], None))
