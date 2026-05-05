"""End-to-end smoke tests against a real upstream provider.

Skipped by default. To run, set ``MAGOS_E2E=1`` and provide whatever upstream
credentials LiteLLM needs (e.g. ``OPENAI_API_KEY``)::

    MAGOS_E2E=1 OPENAI_API_KEY=... uv run pytest -m e2e

The model is configurable via ``MAGOS_E2E_MODEL`` (default ``gpt-4o-mini``);
any LiteLLM-supported model id works.

These tests exercise the full path: FastAPI -> magos.egress.translate -> litellm ->
real provider. They are intentionally minimal because each call costs money.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from fastapi.testclient import TestClient

from magos.ingress.http import create_app

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("MAGOS_E2E") != "1",
        reason="set MAGOS_E2E=1 to run end-to-end provider tests",
    ),
]

MODEL = os.environ.get("MAGOS_E2E_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.environ.get("MAGOS_E2E_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
PROMPT = "Reply with the single word: pong"


def _anthropic_translate_app() -> Any:
    """Build a magos app that forces ``mode: translate`` for /v1/messages.

    The shipped config routes claude-* through byte-exact passthrough,
    which Anthropic's native API rejects when the inbound auth is an OAuth
    access token (``sk-ant-oat*``) without the Claude-Code-only beta
    headers. The translate path goes through ``litellm.anthropic_messages``
    instead, which knows to send OAuth as ``Authorization: Bearer``. Used
    by tool-use tests that need a working Anthropic upstream regardless of
    the inbound key shape.
    """
    from magos.routing import RoutingConfig  # noqa: PLC0415

    cfg = RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/messages"}},
                    "action": {
                        "provider": "anthropic",
                        "mode": "translate",
                        "api_key_env": "ANTHROPIC_API_KEY",
                    },
                }
            ]
        }
    )
    return create_app(routing=cfg)


def _maybe_skip_anthropic_oauth() -> None:
    """Skip if ``ANTHROPIC_API_KEY`` is unset; OAuth is fine for translate path."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")


def _anthropic_inbound_headers() -> dict[str, str]:
    """Headers a Claude-Code-style client sends on /v1/messages.

    The byte-exact passthrough route forwards inbound headers verbatim;
    the Anthropic upstream rejects OAuth tokens unless ``anthropic-beta:
    oauth-2025-04-20`` is present alongside ``anthropic-version`` and the
    bearer. Plain ``sk-ant-api03-*`` keys go via ``x-api-key`` and don't
    need the beta. Empirically verified against api.anthropic.com.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    headers: dict[str, str] = {"anthropic-version": "2023-06-01"}
    if api_key.startswith("sk-ant-oat"):
        headers["Authorization"] = f"Bearer {api_key}"
        headers["anthropic-beta"] = "oauth-2025-04-20"
    elif api_key:
        headers["x-api-key"] = api_key
    return headers


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


def test_openai_non_streaming_real() -> None:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 16,
    }
    with TestClient(create_app()) as client:
        resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["role"] == "assistant"


def test_openai_streaming_real() -> None:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": True,
    }
    with (
        TestClient(create_app()) as client,
        client.stream("POST", "/v1/chat/completions", json=body) as resp,
    ):
        assert resp.status_code == 200
        text = b"".join(resp.iter_bytes()).decode()
    assert "data: [DONE]" in text


def test_anthropic_tool_use_round_trip_real() -> None:
    """Anthropic tool_use round-trip via the translate route.

    Forces ``mode: translate`` so the request takes the
    ``litellm.anthropic_messages`` path with an Anthropic upstream, which
    handles OAuth keys correctly. Cross-provider routing (Anthropic shape
    -> OpenAI upstream) currently hits a LiteLLM tool_choice mapping bug
    that surfaces as a 400 from the upstream Responses API.
    """
    _maybe_skip_anthropic_oauth()
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
    with TestClient(_anthropic_translate_app()) as client:
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


def test_openai_responses_real() -> None:
    """OpenAI Responses shape passes through magos to a real OpenAI upstream.

    Phase A is passthrough-only (no Anthropic <-> Responses translation),
    so this test exercises only the OpenAI Responses -> OpenAI Responses
    path through the routing layer.
    """
    body = {
        "model": "gpt-4o-mini",
        "input": PROMPT,
        "max_output_tokens": 16,
    }
    with TestClient(create_app()) as client:
        resp = client.post("/v1/responses", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data.get("object") == "response"
    output = data.get("output") or []
    assert output, f"expected at least one output item, got {data}"
    text_blocks = [
        c
        for item in output
        if item.get("type") == "message"
        for c in item.get("content", [])
        if c.get("type") == "output_text"
    ]
    assert text_blocks, f"expected output_text content, got {output}"


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
    from magos.routing import RoutingConfig  # noqa: PLC0415

    cfg = RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/messages"}},
                    "action": {
                        "provider": "anthropic",
                        "mode": "translate",
                        "api_key_env": "ANTHROPIC_API_KEY",
                    },
                }
            ]
        }
    )
    body = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": PROMPT}],
    }
    with TestClient(create_app(routing=cfg)) as client:
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


def test_anthropic_passthrough_with_oauth_headers_real() -> None:
    """Byte-exact passthrough of /v1/messages with Claude-Code-style headers.

    Exercises the shipped routing config's claude-* passthrough rule with
    the inbound header set a real Claude Code client sends: OAuth bearer +
    ``anthropic-beta: oauth-2025-04-20`` + ``anthropic-version``. Validates
    that magos forwards them verbatim and the upstream accepts them
    (subject to Extra Usage being enabled on the subscription).
    """
    _maybe_skip_anthropic_oauth()
    body = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": PROMPT}],
    }
    with TestClient(create_app()) as client:
        resp = client.post("/v1/messages", json=body, headers=_anthropic_inbound_headers())
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert data["content"] and data["content"][0]["type"] == "text"


def test_anthropic_passthrough_streaming_with_oauth_headers_real() -> None:
    """Byte-exact passthrough streaming with Claude-Code-style headers.

    Same as the non-streaming sibling but verifies that
    ``passthrough.stream_passthrough`` round-trips the SSE bytes verbatim.
    """
    _maybe_skip_anthropic_oauth()
    body = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": True,
    }
    with (
        TestClient(create_app()) as client,
        client.stream(
            "POST", "/v1/messages", json=body, headers=_anthropic_inbound_headers()
        ) as resp,
    ):
        assert resp.status_code == 200
        text = b"".join(resp.iter_bytes()).decode()
    assert "event: message_start" in text
    assert "event: message_stop" in text


def test_anthropic_streaming_tool_use_real() -> None:
    """Tool definitions survive translated streaming to the Anthropic upstream.

    Asserts the SSE stream contains a ``content_block_start`` with
    ``type=tool_use`` and at least one ``input_json_delta``. Forces
    ``mode: translate`` so the request takes the ``litellm.anthropic_messages``
    streaming path; the bytes that come back are forwarded verbatim by magos.
    """
    _maybe_skip_anthropic_oauth()
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
        TestClient(_anthropic_translate_app()) as client,
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

    Forces ``mode: translate`` so the request takes the
    ``litellm.anthropic_messages`` path with the Anthropic upstream, which
    handles OAuth keys correctly. Validates that magos preserves
    ``tool_use_id`` correlation across turns.
    """
    _maybe_skip_anthropic_oauth()
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
    with TestClient(_anthropic_translate_app()) as client:
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
