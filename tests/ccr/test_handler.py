"""``wrap_response`` handles CCR tool calls in non-streaming responses."""

from __future__ import annotations

import asyncio
from typing import Any

from magos.ccr import wrap_response


def test_wrap_response_passthrough_when_no_ccr_tool_call() -> None:
    """If response has no CCR tool calls, returns it untouched."""
    from magos.egress.translate import TRANSLATE_HANDLERS  # noqa: PLC0415
    from magos.routing.request import RoutedRequest  # noqa: PLC0415

    response = {
        "model": "x",
        "content": [{"type": "text", "text": "hello"}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    req = RoutedRequest(
        endpoint="/v1/messages",
        headers={},
        body={"model": "x", "messages": []},
        raw_body=b"",
    )

    async def fake_completion(**_: Any) -> dict[str, Any]:
        raise AssertionError("continuation should not be invoked")

    out = asyncio.run(
        wrap_response(
            response,
            req=req,
            adapter=TRANSLATE_HANDLERS["/v1/messages"],
            completion=fake_completion,
            dispatch_model="x",
            provider="anthropic",
            forward_headers={},
            api_key=None,
            api_base=None,
        )
    )
    assert out is response


def test_wrap_response_invokes_continuation_for_ccr_tool_call() -> None:
    """If response has a CCR tool call, the handler retrieves and continues."""
    from headroom.cache.compression_store import get_compression_store  # noqa: PLC0415

    from magos.egress.translate import TRANSLATE_HANDLERS  # noqa: PLC0415
    from magos.routing.request import RoutedRequest  # noqa: PLC0415

    # Seed the store so the retrieval has something to find.
    store = get_compression_store()
    hash_key = store.store(
        original='[{"path": "/a"}, {"path": "/b"}]',
        compressed='[{"path": "/a"}]',
        original_tokens=10,
        compressed_tokens=5,
        original_item_count=2,
        compressed_item_count=1,
    )

    initial_response = {
        "model": "claude-sonnet-4-5",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_01",
                "name": "headroom_retrieve",
                "input": {"hash": hash_key},
            }
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    final_response = {
        "model": "claude-sonnet-4-5",
        "content": [{"type": "text", "text": "after retrieval"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 20, "output_tokens": 10},
    }

    call_count = 0

    async def fake_completion(**_: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        return final_response

    req = RoutedRequest(
        endpoint="/v1/messages",
        headers={},
        body={
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "headroom_retrieve"}],
        },
        raw_body=b"",
    )

    out = asyncio.run(
        wrap_response(
            initial_response,
            req=req,
            adapter=TRANSLATE_HANDLERS["/v1/messages"],
            completion=fake_completion,
            dispatch_model="claude-sonnet-4-5",
            provider="anthropic",
            forward_headers={},
            api_key=None,
            api_base=None,
        )
    )

    assert call_count == 1
    # Final response was returned (text content present, not the tool_use).
    assert out.get("stop_reason") == "end_turn" or any(
        block.get("type") == "text" for block in out.get("content", [])
    )


def test_wrap_response_short_circuits_when_not_ccr_request() -> None:
    """If the request didn't carry the CCR tool, even a CCR-shaped response
    in the body shouldn't trigger continuation (defensive)."""
    from magos.egress.translate import TRANSLATE_HANDLERS  # noqa: PLC0415
    from magos.routing.request import RoutedRequest  # noqa: PLC0415

    response = {
        "model": "x",
        "content": [
            {
                "type": "tool_use",
                "id": "x",
                "name": "headroom_retrieve",
                "input": {"hash": "abc"},
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    # Request body has no tools -> not a CCR request.
    req = RoutedRequest(
        endpoint="/v1/messages",
        headers={},
        body={"model": "x", "messages": []},
        raw_body=b"",
    )

    async def fake_completion(**_: Any) -> dict[str, Any]:
        raise AssertionError("continuation should not run")

    out = asyncio.run(
        wrap_response(
            response,
            req=req,
            adapter=TRANSLATE_HANDLERS["/v1/messages"],
            completion=fake_completion,
            dispatch_model="x",
            provider="anthropic",
            forward_headers={},
            api_key=None,
            api_base=None,
        )
    )
    assert out is response
