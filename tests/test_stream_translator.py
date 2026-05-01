"""Unit tests for AnthropicStreamTranslator.

Drives the translator with hand-crafted OpenAI streaming chunks and verifies
the emitted Anthropic event sequence, including text-only flows, tool-use
flows with delta'd JSON arguments, and interleaved text + tool blocks.
"""

from __future__ import annotations

from typing import Any

import pytest

from magos.translation import AnthropicStreamTranslator


def _drain(
    translator: AnthropicStreamTranslator, chunks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for chunk in chunks:
        out.extend(translator.feed(chunk))
    out.extend(translator.finish())
    return out


def _chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


@pytest.mark.unit
def test_text_only_flow() -> None:
    t = AnthropicStreamTranslator(message_id="msg_test")
    events = _drain(
        t,
        [
            _chunk({"role": "assistant"}),
            _chunk({"content": "Hello"}),
            _chunk({"content": " world"}),
            _chunk({}, finish="stop"),
        ],
    )

    types = [e["type"] for e in events]
    assert types == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[0]["message"]["id"] == "msg_test"
    assert events[1]["content_block"] == {"type": "text", "text": ""}
    assert events[2]["delta"] == {"type": "text_delta", "text": "Hello"}
    assert events[3]["delta"] == {"type": "text_delta", "text": " world"}
    assert events[5]["delta"]["stop_reason"] == "end_turn"


@pytest.mark.unit
def test_tool_use_flow_with_delta_json() -> None:
    t = AnthropicStreamTranslator(message_id="msg_tool")
    events = _drain(
        t,
        [
            _chunk({"role": "assistant"}),
            _chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": ""},
                        }
                    ]
                }
            ),
            _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"loc'}}]}),
            _chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'ation":"SF"}'}}]}),
            _chunk({}, finish="tool_calls"),
        ],
    )

    types = [e["type"] for e in events]
    assert types == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    start = events[1]
    assert start["index"] == 0
    assert start["content_block"] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "get_weather",
        "input": {},
    }
    assert events[2]["delta"] == {"type": "input_json_delta", "partial_json": '{"loc'}
    assert events[3]["delta"] == {"type": "input_json_delta", "partial_json": 'ation":"SF"}'}
    assert events[5]["delta"]["stop_reason"] == "tool_use"


@pytest.mark.unit
def test_text_then_tool_use_uses_distinct_indices() -> None:
    t = AnthropicStreamTranslator(message_id="msg_mix")
    events = _drain(
        t,
        [
            _chunk({"role": "assistant"}),
            _chunk({"content": "thinking..."}),
            _chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "do", "arguments": "{}"},
                        }
                    ]
                }
            ),
            _chunk({}, finish="tool_calls"),
        ],
    )

    block_starts = [e for e in events if e["type"] == "content_block_start"]
    block_stops = [e for e in events if e["type"] == "content_block_stop"]
    assert [e["index"] for e in block_starts] == [0, 1]
    assert [e["index"] for e in block_stops] == [0, 1]
    assert block_starts[0]["content_block"]["type"] == "text"
    assert block_starts[1]["content_block"]["type"] == "tool_use"


@pytest.mark.unit
def test_finish_is_idempotent() -> None:
    t = AnthropicStreamTranslator(message_id="msg_x")
    t.feed(_chunk({"content": "hi"}, finish="stop"))
    first = t.finish()
    second = t.finish()
    assert first[-1]["type"] == "message_stop"
    assert second == []


@pytest.mark.unit
def test_usage_tracked_from_final_chunk() -> None:
    t = AnthropicStreamTranslator(message_id="msg_u")
    t.feed(_chunk({"role": "assistant"}))
    t.feed(_chunk({"content": "ok"}))
    t.feed(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17},
        }
    )
    final = t.finish()
    message_delta = next(e for e in final if e["type"] == "message_delta")
    assert message_delta["usage"]["output_tokens"] == 7
