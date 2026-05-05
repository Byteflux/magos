"""Coverage for ``magos.egress.usage``.

Three layers:

- per-shape ``usage_from_*`` extractors against representative response
  bodies, including missing fields / non-dict inputs;
- ``UsageAccumulator`` fed pre-parsed events for each streaming shape
  (Anthropic two-event split, OpenAI Chat terminal-only,
  OpenAI Responses ``response.completed``);
- ``tap_stream`` end-to-end, exercising the SSE byte parser with both
  clean event boundaries and chunks that split mid-event.

Log emissions are asserted via ``structlog.testing.capture_logs``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from structlog.testing import capture_logs

from magos.egress.usage import (
    Usage,
    UsageAccumulator,
    log_usage_from_body,
    shape_for_endpoint,
    tap_stream,
    usage_from_anthropic,
    usage_from_openai_chat,
    usage_from_openai_responses,
)

# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------


def test_usage_from_anthropic_full() -> None:
    body = {
        "model": "claude-sonnet-4-6",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 200,
            "cache_read_input_tokens": 50,
            "cache_creation_input_tokens": 25,
        },
    }
    assert usage_from_anthropic(body) == Usage(input=100, output=200, cache_read=50, cache_write=25)


def test_usage_from_anthropic_missing_fields_default_zero() -> None:
    body = {"usage": {"input_tokens": 7}}
    assert usage_from_anthropic(body) == Usage(input=7)


def test_usage_from_anthropic_garbage_returns_empty() -> None:
    assert usage_from_anthropic({}) == Usage()
    assert usage_from_anthropic({"usage": "nope"}) == Usage()
    assert usage_from_anthropic("not a dict") == Usage()


def test_usage_from_openai_chat_with_cached() -> None:
    body = {
        "usage": {
            "prompt_tokens": 40,
            "completion_tokens": 60,
            "total_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 12},
        }
    }
    assert usage_from_openai_chat(body) == Usage(input=40, output=60, cache_read=12)


def test_usage_from_openai_chat_no_details() -> None:
    body = {"usage": {"prompt_tokens": 1, "completion_tokens": 2}}
    assert usage_from_openai_chat(body) == Usage(input=1, output=2)


def test_usage_from_openai_responses_with_cached() -> None:
    body = {
        "usage": {
            "input_tokens": 80,
            "output_tokens": 160,
            "input_tokens_details": {"cached_tokens": 40},
        }
    }
    assert usage_from_openai_responses(body) == Usage(input=80, output=160, cache_read=40)


def test_safe_int_rejects_negative_and_non_int() -> None:
    body = {"usage": {"input_tokens": -5, "output_tokens": "10"}}
    assert usage_from_anthropic(body) == Usage()


# ---------------------------------------------------------------------------
# Accumulators
# ---------------------------------------------------------------------------


def test_accumulator_anthropic_combines_message_start_and_delta() -> None:
    acc = UsageAccumulator("anthropic")
    acc.feed(
        "message_start",
        {
            "message": {
                "model": "claude-sonnet-4-6",
                "usage": {
                    "input_tokens": 100,
                    "cache_read_input_tokens": 30,
                    "cache_creation_input_tokens": 10,
                },
            }
        },
    )
    acc.feed("message_delta", {"usage": {"output_tokens": 250}})
    assert acc.snapshot() == Usage(input=100, output=250, cache_read=30, cache_write=10)
    assert acc.model == "claude-sonnet-4-6"


def test_accumulator_openai_chat_takes_terminal_chunk() -> None:
    acc = UsageAccumulator("openai-chat")
    # Earlier chunks have no usage field.
    acc.feed(None, {"choices": [{"delta": {"content": "h"}}]})
    acc.feed(
        None,
        {
            "model": "gpt-4o",
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 7,
                "prompt_tokens_details": {"cached_tokens": 2},
            },
        },
    )
    assert acc.snapshot() == Usage(input=5, output=7, cache_read=2)
    assert acc.model == "gpt-4o"


def test_accumulator_openai_responses_uses_response_completed() -> None:
    acc = UsageAccumulator("openai-responses")
    acc.feed(
        "response.completed",
        {
            "response": {
                "model": "gpt-4o",
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 22,
                    "input_tokens_details": {"cached_tokens": 3},
                },
            }
        },
    )
    assert acc.snapshot() == Usage(input=11, output=22, cache_read=3)


# ---------------------------------------------------------------------------
# log_usage_from_body
# ---------------------------------------------------------------------------


def test_log_usage_emits_canonical_event() -> None:
    body = {
        "model": "claude-sonnet-4-6",
        "usage": {
            "input_tokens": 1,
            "output_tokens": 2,
            "cache_read_input_tokens": 3,
            "cache_creation_input_tokens": 4,
        },
    }
    with capture_logs() as logs:
        log_usage_from_body("anthropic", body, endpoint="/v1/messages")
    matches = [e for e in logs if e.get("event") == "egress.usage"]
    assert len(matches) == 1
    entry = matches[0]
    assert entry["shape"] == "anthropic"
    assert entry["endpoint"] == "/v1/messages"
    assert entry["model"] == "claude-sonnet-4-6"
    assert entry["input"] == 1
    assert entry["output"] == 2
    assert entry["cache_read"] == 3
    assert entry["cache_write"] == 4


def test_log_usage_skips_empty_payload() -> None:
    with capture_logs() as logs:
        log_usage_from_body("openai-chat", {"model": "gpt-4o"}, endpoint="/v1/chat/completions")
    assert [e for e in logs if e.get("event") == "egress.usage"] == []


# ---------------------------------------------------------------------------
# shape_for_endpoint
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# tap_stream: end-to-end SSE
# ---------------------------------------------------------------------------


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
    chunks: list[bytes], shape: str, endpoint: str
) -> tuple[list[bytes], list[dict[str, Any]]]:
    forwarded: list[bytes] = []
    with capture_logs() as logs:
        async for chunk in tap_stream(_bytes_iter(chunks), shape, endpoint=endpoint):  # type: ignore[arg-type]
            forwarded.append(chunk)
    return forwarded, [dict(e) for e in logs if e.get("event") == "egress.usage"]


def test_tap_stream_anthropic_logs_combined_usage() -> None:
    chunks = [
        _anthropic_message_start(),
        _anthropic_message_delta(),
        _anthropic_message_stop(),
    ]
    forwarded, matches = asyncio.run(_drain(chunks, "anthropic", "/v1/messages"))
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
    _, matches = asyncio.run(_drain(chunks, "openai-chat", "/v1/chat/completions"))
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
    _, matches = asyncio.run(_drain([chunk], "openai-responses", "/v1/responses"))
    assert len(matches) == 1
    assert matches[0]["input"] == 11
    assert matches[0]["output"] == 22
    assert matches[0]["cache_read"] == 3


def test_tap_stream_handles_event_split_across_chunks() -> None:
    """An event spanning two byte chunks must still parse."""
    raw = _anthropic_message_start() + _anthropic_message_delta()
    split_at = len(raw) // 2
    chunks = [raw[:split_at], raw[split_at:]]
    _, matches = asyncio.run(_drain(chunks, "anthropic", "/v1/messages"))
    assert len(matches) == 1
    assert matches[0]["input"] == 100
    assert matches[0]["output"] == 250


def test_tap_stream_no_usage_emits_no_log() -> None:
    """Streams that don't carry usage (e.g. include_usage off) stay silent."""
    chunks = [b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n', b"data: [DONE]\n\n"]
    _, matches = asyncio.run(_drain(chunks, "openai-chat", "/v1/chat/completions"))
    assert matches == []


def test_tap_stream_flushes_trailing_event_without_blank_line() -> None:
    """Some upstreams omit the trailing \\n\\n on the final event.

    First event has the proper ``\\n\\n`` separator so the stream parser
    consumes it normally; the final event is missing its trailing blank
    line and only gets flushed by the ``finally`` branch.
    """
    head = _anthropic_message_start()  # well-formed, ends with \n\n
    tail = _anthropic_message_delta().rstrip(b"\n")  # truncated terminator
    _, matches = asyncio.run(_drain([head, tail], "anthropic", "/v1/messages"))
    assert len(matches) == 1
    entry = matches[0]
    assert entry["input"] == 100
    assert entry["output"] == 250


# ---------------------------------------------------------------------------
# Stream-tap robustness
# ---------------------------------------------------------------------------


def test_tap_stream_ignores_non_json_data_lines() -> None:
    """``data: [DONE]`` and other non-JSON sentinels must not crash the parser."""
    _, matches = asyncio.run(_drain([b"data: [DONE]\n\n"], "openai-chat", "/v1/chat/completions"))
    assert matches == []


def test_usage_is_empty_property() -> None:
    assert Usage().is_empty
    assert not Usage(input=1).is_empty


def test_usage_accumulator_ignores_unhandled_events() -> None:
    acc = UsageAccumulator("anthropic")
    acc.feed("ping", {"foo": "bar"})
    acc.feed("content_block_start", {"type": "content_block_start"})
    assert acc.snapshot() == Usage()


def _payload_with_no_usage() -> dict[str, Any]:
    return {"model": "x"}


def test_extractors_handle_missing_usage_block() -> None:
    assert usage_from_anthropic(_payload_with_no_usage()) == Usage()
    assert usage_from_openai_chat(_payload_with_no_usage()) == Usage()
    assert usage_from_openai_responses(_payload_with_no_usage()) == Usage()
