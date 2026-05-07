"""`tap_stream` end-to-end SSE byte parser + accumulator integration."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from structlog.testing import capture_logs

from magos.dispatch.usage import tap_stream
from magos.shapes import ANTHROPIC, OPENAI_CHAT, OPENAI_RESPONSES, Shape


async def _bytes_iter(chunks: list[bytes]) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


def _anthropic_message_start() -> bytes:
    payload = {
        "type": "message_start",
        "message": {
            "model": "claude-sonnet-4-6",
            "usage": {
                "input_tokens": 100,
                "cache_read_input_tokens": 30,
                "cache_creation_input_tokens": 10,
            },
        },
    }
    return f"event: message_start\ndata: {json.dumps(payload)}\n\n".encode()


def _anthropic_message_delta() -> bytes:
    payload = {"type": "message_delta", "usage": {"output_tokens": 250}}
    return f"event: message_delta\ndata: {json.dumps(payload)}\n\n".encode()


def _anthropic_message_stop() -> bytes:
    return b"event: message_stop\ndata: {}\n\n"


async def _drain(
    chunks: list[bytes], shape: Shape, endpoint: str
) -> tuple[list[bytes], list[dict[str, Any]]]:
    forwarded: list[bytes] = []
    with capture_logs() as logs:
        async for chunk in tap_stream(_bytes_iter(chunks), shape, endpoint=endpoint):
            forwarded.append(chunk)
    return forwarded, [dict(e) for e in logs if e.get("event") == "egress.usage"]


def test_tap_stream_anthropic_logs_combined_usage() -> None:
    chunks = [
        _anthropic_message_start(),
        _anthropic_message_delta(),
        _anthropic_message_stop(),
    ]
    forwarded, matches = asyncio.run(_drain(chunks, ANTHROPIC, "/v1/messages"))
    assert b"".join(forwarded) == b"".join(chunks)
    assert len(matches) == 1
    entry = matches[0]
    assert entry["input"] == 100
    assert entry["output"] == 250
    assert entry["cache_read"] == 30
    assert entry["cache_write"] == 10
    assert entry["stream"] is True
    assert entry["model"] == "claude-sonnet-4-6"


def test_tap_stream_openai_chat_terminal_chunk() -> None:
    chunks = [
        b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
        (
            b'data: {"model":"gpt-4o","usage":{"prompt_tokens":5,"completion_tokens":7,'
            b'"prompt_tokens_details":{"cached_tokens":2}}}\n\n'
        ),
        b"data: [DONE]\n\n",
    ]
    _, matches = asyncio.run(_drain(chunks, OPENAI_CHAT, "/v1/chat/completions"))
    assert len(matches) == 1
    entry = matches[0]
    assert entry["input"] == 5
    assert entry["output"] == 7
    assert entry["cache_read"] == 2
    assert entry["model"] == "gpt-4o"


def test_tap_stream_openai_responses_completed() -> None:
    completed = {
        "type": "response.completed",
        "response": {
            "model": "gpt-4o",
            "usage": {
                "input_tokens": 11,
                "output_tokens": 22,
                "input_tokens_details": {"cached_tokens": 3},
            },
        },
    }
    chunk = f"event: response.completed\ndata: {json.dumps(completed)}\n\n".encode()
    _, matches = asyncio.run(_drain([chunk], OPENAI_RESPONSES, "/v1/responses"))
    assert len(matches) == 1
    assert matches[0]["input"] == 11
    assert matches[0]["output"] == 22
    assert matches[0]["cache_read"] == 3


def test_tap_stream_handles_event_split_across_chunks() -> None:
    """An event spanning two byte chunks must still parse."""
    raw = _anthropic_message_start() + _anthropic_message_delta()
    split_at = len(raw) // 2
    chunks = [raw[:split_at], raw[split_at:]]
    _, matches = asyncio.run(_drain(chunks, ANTHROPIC, "/v1/messages"))
    assert len(matches) == 1
    assert matches[0]["input"] == 100
    assert matches[0]["output"] == 250


def test_tap_stream_no_usage_emits_no_log() -> None:
    """Streams that don't carry usage (e.g. include_usage off) stay silent."""
    chunks = [b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n', b"data: [DONE]\n\n"]
    _, matches = asyncio.run(_drain(chunks, OPENAI_CHAT, "/v1/chat/completions"))
    assert matches == []


def test_tap_stream_flushes_trailing_event_without_blank_line() -> None:
    """Some upstreams omit the trailing `\\n\\n` on the final event.

    First event has the proper `\\n\\n` separator so the stream parser
    consumes it normally; the final event is missing its trailing blank
    line and only gets flushed by the `finally` branch.
    """
    head = _anthropic_message_start()  # well-formed, ends with \n\n
    tail = _anthropic_message_delta().rstrip(b"\n")  # truncated terminator
    _, matches = asyncio.run(_drain([head, tail], ANTHROPIC, "/v1/messages"))
    assert len(matches) == 1
    entry = matches[0]
    assert entry["input"] == 100
    assert entry["output"] == 250


def test_tap_stream_ignores_non_json_data_lines() -> None:
    """`data: [DONE]` and other non-JSON sentinels must not crash the parser."""
    _, matches = asyncio.run(_drain([b"data: [DONE]\n\n"], OPENAI_CHAT, "/v1/chat/completions"))
    assert matches == []


def test_tap_stream_fires_on_complete_after_final_usage() -> None:
    async def _run() -> tuple[bytes, list[Any]]:
        async def upstream() -> AsyncIterator[bytes]:
            yield (
                b'event: message_start\ndata: {"message": {"usage": '
                b'{"input_tokens": 100, "output_tokens": 0, "cache_read_input_tokens": 0, '
                b'"cache_creation_input_tokens": 0}, "model": "claude-x"}}\n\n'
            )
            yield b'event: message_delta\ndata: {"usage": {"output_tokens": 50}}\n\n'

        seen: list[Any] = []
        bytes_out = b""
        async for chunk in tap_stream(
            upstream(), ANTHROPIC, endpoint="/v1/messages", on_complete=seen.append
        ):
            bytes_out += chunk
        return bytes_out, seen

    bytes_out, seen = asyncio.run(_run())
    assert b"message_start" in bytes_out
    assert len(seen) == 1
    assert seen[0].input == 100
    assert seen[0].output == 50
