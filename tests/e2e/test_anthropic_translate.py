"""End-to-end smoke tests for the Anthropic translate path.

Covers Anthropic-shape requests against both upstream shapes (translated
to/from OpenAI by LiteLLM), tool-use round-trips, count_tokens, and the
unmatched-route 404 envelope. See ``tests/e2e/conftest.py`` for the
``MAGOS_E2E=1`` skip gate.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from magos.api import create_app

from ._helpers import (
    ANTHROPIC_MODEL,
    MODEL,
    PROMPT,
    anthropic_translate_app,
    maybe_skip_anthropic_oauth,
)


def test_anthropic_non_streaming_real() -> None:
    body = {
        "model": MODEL,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": PROMPT}],
    }
    with TestClient(create_app()) as client:
        resp = client.post("/v1/messages", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert data["content"] and data["content"][0]["type"] == "text"


def test_anthropic_streaming_real() -> None:
    body = {
        "model": MODEL,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": True,
    }
    with (
        TestClient(create_app()) as client,
        client.stream("POST", "/v1/messages", json=body) as resp,
    ):
        assert resp.status_code == 200
        text = b"".join(resp.iter_bytes()).decode()
    assert "event: message_start" in text
    assert "event: message_stop" in text


def test_anthropic_tool_use_round_trip_real() -> None:
    """Anthropic tool_use round-trip via the translate route.

    Forces ``gateway: translate`` so the request takes the
    ``litellm.anthropic_messages`` path with an Anthropic upstream, which
    handles OAuth keys correctly. Cross-provider routing (Anthropic shape
    -> OpenAI upstream) currently hits a LiteLLM tool_choice mapping bug
    that surfaces as a 400 from the upstream Responses API.
    """
    maybe_skip_anthropic_oauth()
    body = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 64,
        "tools": [
            {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
        "tool_choice": {"type": "any"},
        "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
    }
    with TestClient(anthropic_translate_app()) as client:
        resp = client.post("/v1/messages", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["stop_reason"] == "tool_use"
    tool_uses = [b for b in data["content"] if b.get("type") == "tool_use"]
    assert tool_uses, f"expected tool_use block, got {data['content']}"
    assert tool_uses[0]["name"] == "get_weather"
    assert "city" in tool_uses[0]["input"]


def test_anthropic_multi_turn_with_system_real() -> None:
    """System prompt + multi-turn history survive translation."""
    body = {
        "model": MODEL,
        "max_tokens": 16,
        "system": "You always reply with exactly one word, lowercase, no punctuation.",
        "messages": [
            {"role": "user", "content": "Say hello."},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "Now say goodbye."},
        ],
    }
    with TestClient(create_app()) as client:
        resp = client.post("/v1/messages", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    text_blocks = [b for b in data["content"] if b.get("type") == "text"]
    assert text_blocks, data
    reply = text_blocks[0]["text"].strip().lower()
    assert "goodbye" in reply or "bye" in reply, f"expected farewell, got {reply!r}"


def test_unmatched_route_returns_404_anthropic_envelope_real() -> None:
    """Live request that no rule matches returns a 404 in Anthropic shape."""
    body = {
        "model": "no-such-model-anywhere",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": PROMPT}],
    }
    with TestClient(create_app()) as client:
        resp = client.post("/v1/messages", json=body)
    assert resp.status_code == 404, resp.text
    payload = resp.json()
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "not_found_error"
    assert "no-such-model-anywhere" in payload["error"]["message"]


def test_anthropic_count_tokens_real() -> None:
    """count_tokens returns a positive count via litellm.acount_tokens."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hello world, count me."}],
    }
    with TestClient(create_app()) as client:
        resp = client.post("/v1/messages/count_tokens", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert isinstance(data["input_tokens"], int)
    assert data["input_tokens"] > 0


def test_anthropic_shape_anthropic_upstream_translated_real() -> None:
    """Anthropic-shape request -> Anthropic upstream via the translated path.

    The shipped magos.example.yaml routes claude-* to passthrough; this
    test injects an alternative routing config that forces ``mode:
    translate`` so the request takes the litellm.anthropic_messages
    code path with a real Anthropic upstream.
    """
    body = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": PROMPT}],
    }
    with TestClient(anthropic_translate_app()) as client:
        resp = client.post("/v1/messages", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert data["content"] and data["content"][0]["type"] == "text"


def test_openai_shape_anthropic_upstream_real() -> None:
    """OpenAI-shape request -> Anthropic upstream via litellm.

    LiteLLM owns the OpenAI -> Anthropic conversion here; magos contributes
    provider-prefix resolution and header forwarding. Closes the cross-
    direction cell of the (shape x upstream) matrix.
    """
    body = {
        "model": ANTHROPIC_MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 16,
    }
    with TestClient(create_app()) as client:
        resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["role"] == "assistant"


def test_anthropic_count_tokens_anthropic_native_real() -> None:
    """count_tokens for a claude-* model via litellm's native API call.

    LiteLLM's ``acount_tokens`` auto-selects the upstream's native
    count-tokens endpoint for ``anthropic/`` models. OAuth tokens are
    accepted on this endpoint when the subscription has Extra Usage
    enabled; otherwise a separate ``sk-ant-api03-*`` key is required.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    body = {
        "model": ANTHROPIC_MODEL,
        "messages": [{"role": "user", "content": "Hello world, count me."}],
    }
    with TestClient(create_app()) as client:
        resp = client.post("/v1/messages/count_tokens", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert isinstance(data["input_tokens"], int)
    assert data["input_tokens"] > 0


def test_anthropic_streaming_tool_use_real() -> None:
    """Tool definitions survive translated streaming to the Anthropic upstream.

    Asserts the SSE stream contains a ``content_block_start`` with
    ``type=tool_use`` and at least one ``input_json_delta``. Forces
    ``gateway: translate`` so the request takes the ``litellm.anthropic_messages``
    streaming path; the bytes that come back are forwarded verbatim by magos.
    """
    maybe_skip_anthropic_oauth()
    body = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 64,
        "stream": True,
        "tools": [
            {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
        "tool_choice": {"type": "any"},
        "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
    }
    with (
        TestClient(anthropic_translate_app()) as client,
        client.stream("POST", "/v1/messages", json=body) as resp,
    ):
        assert resp.status_code == 200
        text = b"".join(resp.iter_bytes()).decode()
    assert "event: message_start" in text
    assert '"type": "tool_use"' in text or '"type":"tool_use"' in text
    assert "input_json_delta" in text
    assert "event: message_stop" in text


def test_anthropic_tool_result_followup_real() -> None:
    """Full agent-loop turn: tool_use -> tool_result -> final text.

    Forces ``gateway: translate`` so the request takes the
    ``litellm.anthropic_messages`` path with the Anthropic upstream, which
    handles OAuth keys correctly. Validates that magos preserves
    ``tool_use_id`` correlation across turns.
    """
    maybe_skip_anthropic_oauth()
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]
    first = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 64,
        "tools": tools,
        "tool_choice": {"type": "any"},
        "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
    }
    with TestClient(anthropic_translate_app()) as client:
        resp = client.post("/v1/messages", json=first)
        assert resp.status_code == 200, resp.text
        first_data = resp.json()
        tool_uses = [b for b in first_data["content"] if b.get("type") == "tool_use"]
        assert tool_uses, f"expected tool_use block, got {first_data['content']}"
        tool_use = tool_uses[0]

        followup = {
            "model": ANTHROPIC_MODEL,
            "max_tokens": 64,
            "tools": tools,
            "messages": [
                {"role": "user", "content": "What's the weather in Tokyo?"},
                {"role": "assistant", "content": first_data["content"]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use["id"],
                            "content": "Sunny, 22 degrees Celsius.",
                        }
                    ],
                },
            ],
        }
        resp = client.post("/v1/messages", json=followup)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    text_blocks = [b for b in data["content"] if b.get("type") == "text"]
    assert text_blocks, data
    assert text_blocks[0]["text"].strip(), "expected non-empty final text"
