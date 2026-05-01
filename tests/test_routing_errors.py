"""Tests for ``magos.routing.errors`` envelope shaping."""

from __future__ import annotations

import dataclasses

import pytest

from magos.routing.errors import (
    RouteError,
    error_envelope,
    format_dispatch_error_message,
    format_unmatched_message,
)


def test_route_error_is_frozen() -> None:
    err = RouteError(
        status=404,
        code="unmatched",
        message="x",
        model="claude-3",
        endpoint="/v1/messages",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        err.status = 503  # type: ignore[misc]


def test_anthropic_envelope_for_messages_unmatched() -> None:
    body = error_envelope(endpoint="/v1/messages", code="unmatched", message="no match")
    assert body == {
        "type": "error",
        "error": {"type": "not_found_error", "message": "no match"},
    }


def test_anthropic_envelope_for_count_tokens_dispatch_error() -> None:
    body = error_envelope(
        endpoint="/v1/messages/count_tokens",
        code="dispatch_error",
        message="missing key",
    )
    assert body == {
        "type": "error",
        "error": {"type": "api_error", "message": "missing key"},
    }


def test_openai_envelope_for_chat_completions_unmatched() -> None:
    body = error_envelope(endpoint="/v1/chat/completions", code="unmatched", message="no match")
    assert body == {
        "error": {
            "message": "no match",
            "type": "invalid_request_error",
            "code": "no_route_matched",
        }
    }


def test_openai_envelope_for_chat_completions_dispatch_error() -> None:
    body = error_envelope(
        endpoint="/v1/chat/completions",
        code="dispatch_error",
        message="missing key",
    )
    assert body["error"]["type"] == "server_error"
    assert body["error"]["code"] == "dispatch_error"


def test_unmatched_message_includes_model() -> None:
    msg = format_unmatched_message("claude-3-5-sonnet-20241022")
    assert "claude-3-5-sonnet-20241022" in msg
    assert "magos.yaml" in msg


def test_unmatched_message_omits_quoted_empty() -> None:
    # When model is missing, do not render ``''`` since that's confusing.
    msg = format_unmatched_message("")
    assert "''" not in msg
    assert "magos.yaml" in msg


def test_dispatch_error_message_does_not_echo_env_var_names() -> None:
    # Reason text comes from caller; helper should not silently amend it.
    msg = format_dispatch_error_message("missing api key")
    assert msg == "route configuration error: missing api key"
