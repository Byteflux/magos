"""End-to-end smoke tests against a real upstream provider.

Skipped by default. To run, set ``MAGOS_E2E=1`` and provide whatever upstream
credentials LiteLLM needs (e.g. ``OPENAI_API_KEY``)::

    MAGOS_E2E=1 OPENAI_API_KEY=... uv run pytest -m e2e

The model is configurable via ``MAGOS_E2E_MODEL`` (default ``gpt-4o-mini``);
any LiteLLM-supported model id works.

These tests exercise the full path: FastAPI -> magos.proxy -> litellm ->
real provider. They are intentionally minimal because each call costs money.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from magos.server import create_app

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
    """Tool definition (Anthropic) -> tool_calls (OpenAI) -> tool_use (Anthropic)."""
    body = {
        "model": MODEL,
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
    with TestClient(create_app()) as client:
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
    """count_tokens returns a positive estimate for an OpenAI model (local path)."""
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
    translate`` so the request takes the forward.py + litellm + reverse.py
    round-trip with a real Anthropic upstream.
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


def test_anthropic_count_tokens_anthropic_passthrough_real() -> None:
    """count_tokens via the Anthropic native passthrough endpoint.

    A claude-* model on /v1/messages/count_tokens hits the rule with
    ``count_tokens_mode: passthrough``, which calls
    ``anthropic.AsyncAnthropic().messages.count_tokens`` via the SDK. The
    SDK reads ``ANTHROPIC_API_KEY`` from the env and sends it as
    ``x-api-key``; OAuth tokens (``sk-ant-oat*``) are rejected by this
    endpoint, so we skip in that case rather than fail spuriously.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key.startswith("sk-ant-oat"):
        pytest.skip(
            "ANTHROPIC_API_KEY is an OAuth access token; count_tokens API "
            "requires a regular sk-ant-api03-* key"
        )
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
    """Tool definitions survive translation in streaming mode.

    Asserts the SSE stream contains a ``content_block_start`` with
    ``type=tool_use`` and at least one ``input_json_delta``, the part of
    ``streaming.py`` not exercised by ``test_anthropic_streaming_real``.
    """
    body = {
        "model": MODEL,
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
        TestClient(create_app()) as client,
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

    Sends an initial tools-enabled request, captures the ``tool_use`` block,
    then sends a follow-up turn with the assistant's tool_use plus a
    matching ``tool_result`` user message, and asserts the model produces a
    final text response. Validates that translation preserves tool_use_id
    correlation across turns.
    """
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
        "model": MODEL,
        "max_tokens": 64,
        "tools": tools,
        "tool_choice": {"type": "any"},
        "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
    }
    with TestClient(create_app()) as client:
        resp = client.post("/v1/messages", json=first)
        assert resp.status_code == 200, resp.text
        first_data = resp.json()
        tool_uses = [b for b in first_data["content"] if b.get("type") == "tool_use"]
        assert tool_uses, f"expected tool_use block, got {first_data['content']}"
        tool_use = tool_uses[0]

        followup = {
            "model": MODEL,
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
