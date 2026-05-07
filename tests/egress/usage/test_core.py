"""Coverage for ``magos.egress.usage.core``: extractor + log helpers."""

from __future__ import annotations

from typing import Any

from structlog.testing import capture_logs

from magos.egress.usage import (
    Usage,
    log_usage_from_body,
    usage_from_body,
)

# --- usage_from_body (generic, reads shape.usage_keys) ---


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
    assert usage_from_body("anthropic", body) == Usage(
        input=100, output=200, cache_read=50, cache_write=25
    )


def test_usage_from_anthropic_missing_fields_default_zero() -> None:
    body = {"usage": {"input_tokens": 7}}
    assert usage_from_body("anthropic", body) == Usage(input=7)


def test_usage_from_anthropic_garbage_returns_empty() -> None:
    assert usage_from_body("anthropic", {}) == Usage()
    assert usage_from_body("anthropic", {"usage": "nope"}) == Usage()
    assert usage_from_body("anthropic", "not a dict") == Usage()


def test_usage_from_openai_chat_with_cached() -> None:
    body = {
        "usage": {
            "prompt_tokens": 40,
            "completion_tokens": 60,
            "total_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 12},
        }
    }
    assert usage_from_body("openai-chat", body) == Usage(input=40, output=60, cache_read=12)


def test_usage_from_openai_chat_no_details() -> None:
    body = {"usage": {"prompt_tokens": 1, "completion_tokens": 2}}
    assert usage_from_body("openai-chat", body) == Usage(input=1, output=2)


def test_usage_from_openai_responses_with_cached() -> None:
    body = {
        "usage": {
            "input_tokens": 80,
            "output_tokens": 160,
            "input_tokens_details": {"cached_tokens": 40},
        }
    }
    assert usage_from_body("openai-responses", body) == Usage(input=80, output=160, cache_read=40)


def test_safe_int_rejects_negative_and_non_int() -> None:
    body = {"usage": {"input_tokens": -5, "output_tokens": "10"}}
    assert usage_from_body("anthropic", body) == Usage()


def test_extractors_handle_missing_usage_block() -> None:
    payload = {"model": "x"}
    assert usage_from_body("anthropic", payload) == Usage()
    assert usage_from_body("openai-chat", payload) == Usage()
    assert usage_from_body("openai-responses", payload) == Usage()


def test_openai_shapes_omit_cache_write() -> None:
    """``cache_write`` is Anthropic-only; OpenAI shapes ignore the field."""
    body = {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    assert usage_from_body("openai-chat", body).cache_write == 0


# --- log_usage_from_body ---


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


def test_log_usage_from_body_returns_usage() -> None:
    body = {"model": "x", "usage": {"input_tokens": 100, "output_tokens": 50}}
    result = log_usage_from_body("anthropic", body, endpoint="/v1/messages")
    assert result.input == 100
    assert result.output == 50


def test_log_usage_from_body_fires_on_complete_callback() -> None:
    seen: list[Any] = []
    body = {"model": "x", "usage": {"input_tokens": 100, "output_tokens": 50}}
    log_usage_from_body("anthropic", body, endpoint="/v1/messages", on_complete=seen.append)
    assert len(seen) == 1
    assert seen[0].input == 100


def test_log_usage_from_body_skips_callback_on_empty_usage() -> None:
    seen: list[Any] = []
    log_usage_from_body(
        "anthropic", {"model": "x"}, endpoint="/v1/messages", on_complete=seen.append
    )
    assert seen == []


# --- Usage dataclass ---


def test_usage_is_empty_property() -> None:
    assert Usage().is_empty
    assert not Usage(input=1).is_empty
