"""`UsageAccumulator` fed pre-parsed events for each streaming shape."""

from __future__ import annotations

from magos.dispatch.usage import Usage, UsageAccumulator
from magos.shapes import ANTHROPIC, OPENAI_CHAT, OPENAI_RESPONSES


def test_accumulator_anthropic_combines_message_start_and_delta() -> None:
    acc = UsageAccumulator(ANTHROPIC)
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
    acc = UsageAccumulator(OPENAI_CHAT)
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
    acc = UsageAccumulator(OPENAI_RESPONSES)
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


def test_usage_accumulator_ignores_unhandled_events() -> None:
    acc = UsageAccumulator(ANTHROPIC)
    acc.feed("ping", {"foo": "bar"})
    acc.feed("content_block_start", {"type": "content_block_start"})
    assert acc.snapshot() == Usage()
