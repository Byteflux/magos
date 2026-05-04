"""Shared helpers for ingress HTTP tests.

Each test injects a routing config via ``create_app(routing=...)`` and
overrides the matching completion dependency so no real upstream is
contacted. Passthrough wire behaviour itself is unit-tested in
``tests/egress/test_passthrough.py``; the helpers here just stand up a
TestClient with the right rules and stubs in place.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from fastapi.testclient import TestClient

from magos.ingress.http import create_app
from magos.ingress.http.handlers import (
    get_anthropic_messages_completion,
    get_completion,
    get_count_tokens_completion,
    get_responses_completion,
)
from magos.routing import RoutingConfig


def translate_only_cfg(provider: str = "openai") -> RoutingConfig:
    """A minimal config where every endpoint translates through litellm.

    Used by the bulk of the server tests so the existing seam (a faked
    completion callable) keeps exercising the same code paths.
    """
    return RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/messages"}},
                    "action": {"provider": provider, "mode": "translate"},
                },
                {
                    "match": {"endpoint": {"literal": "/v1/chat/completions"}},
                    "action": {"provider": provider, "mode": "translate"},
                },
                {
                    "match": {"endpoint": {"literal": "/v1/messages/count_tokens"}},
                    "action": {"provider": provider, "mode": "translate"},
                },
            ]
        }
    )


def anthropic_translate_cfg() -> RoutingConfig:
    return RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/messages"}},
                    "action": {"provider": "anthropic", "mode": "translate"},
                }
            ]
        }
    )


def client_with(
    *,
    chat_completion: Any | None = None,
    anthropic_completion: Any | None = None,
    count_tokens_completion: Any | None = None,
    responses_completion: Any | None = None,
    routing: RoutingConfig | None = None,
) -> Iterator[TestClient]:
    cfg = routing if routing is not None else translate_only_cfg()
    app = create_app(routing=cfg)
    if chat_completion is not None:
        app.dependency_overrides[get_completion] = lambda: chat_completion
    if anthropic_completion is not None:
        app.dependency_overrides[get_anthropic_messages_completion] = lambda: anthropic_completion
    if count_tokens_completion is not None:
        app.dependency_overrides[get_count_tokens_completion] = lambda: count_tokens_completion
    if responses_completion is not None:
        app.dependency_overrides[get_responses_completion] = lambda: responses_completion
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


ANTHROPIC_MESSAGE_RESPONSE: dict[str, Any] = {
    "id": "msg_01ABC",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku",
    "content": [{"type": "text", "text": "hello"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
}

OPENAI_CHAT_RESPONSE: dict[str, Any] = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
