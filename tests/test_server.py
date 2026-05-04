"""FastAPI endpoint tests for the magos server.

Drives the server with TestClient. Each test injects a routing config via
``create_app(routing=...)`` and overrides the matching completion
dependency so no real upstream is contacted. Passthrough tests build a
minimal config that forces the matched rule's mode; passthrough wire
behavior itself is unit-tested in ``test_passthrough.py``.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from magos.routing import RoutingConfig
from magos.server import (
    create_app,
    get_anthropic_messages_completion,
    get_completion,
    get_count_tokens_completion,
    get_responses_completion,
)


def _translate_only_cfg(provider: str = "openai") -> RoutingConfig:
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


def _client_with(
    *,
    chat_completion: Any | None = None,
    anthropic_completion: Any | None = None,
    count_tokens_completion: Any | None = None,
    responses_completion: Any | None = None,
    routing: RoutingConfig | None = None,
) -> Iterator[TestClient]:
    cfg = routing if routing is not None else _translate_only_cfg()
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


_ANTHROPIC_MESSAGE_RESPONSE: dict[str, Any] = {
    "id": "msg_01ABC",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku",
    "content": [{"type": "text", "text": "hello"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
}

_OPENAI_CHAT_RESPONSE: dict[str, Any] = {
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


def _anthropic_translate_cfg() -> RoutingConfig:
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


@pytest.mark.integration
def test_messages_endpoint_dispatches_to_anthropic_messages() -> None:
    """/v1/messages translate-mode goes through litellm.anthropic_messages.

    The fake stand-in for ``litellm.anthropic_messages`` records its
    kwargs and returns an Anthropic-shape body; magos forwards it verbatim.
    """
    received: dict[str, Any] = {}

    async def fake_anthropic(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return _ANTHROPIC_MESSAGE_RESPONSE

    body = {
        "model": "claude-haiku",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    }
    for client in _client_with(
        anthropic_completion=fake_anthropic, routing=_anthropic_translate_cfg()
    ):
        resp = client.post("/v1/messages", json=body)

    assert resp.status_code == 200
    assert resp.json() == _ANTHROPIC_MESSAGE_RESPONSE
    # dispatch_model gets the anthropic/ prefix; max_tokens + messages survive.
    assert received["model"] == "anthropic/claude-haiku"
    assert received["max_tokens"] == 16
    assert received["messages"] == body["messages"]


@pytest.mark.integration
def test_chat_completions_endpoint_passes_through() -> None:
    openai_request = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
    }
    received: dict[str, Any] = {}

    async def fake_completion(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return _OPENAI_CHAT_RESPONSE

    for client in _client_with(chat_completion=fake_completion):
        resp = client.post("/v1/chat/completions", json=openai_request)

    assert resp.status_code == 200
    assert resp.json() == _OPENAI_CHAT_RESPONSE
    expected = {**openai_request, "model": f"openai/{openai_request['model']}"}
    received_no_headers = {k: v for k, v in received.items() if k != "extra_headers"}
    assert received_no_headers == expected
    assert "extra_headers" in received


@pytest.mark.integration
def test_messages_streams_anthropic_bytes_verbatim() -> None:
    """LiteLLM's anthropic_messages streaming yields raw Anthropic SSE bytes."""
    chunks = [
        b'event: message_start\ndata: {"type": "message_start"}\n\n',
        b'event: content_block_delta\ndata: {"type": "content_block_delta", '
        b'"delta": {"type": "text_delta", "text": "hi"}}\n\n',
        b'event: message_stop\ndata: {"type": "message_stop"}\n\n',
    ]

    async def fake_iter() -> Any:
        for chunk in chunks:
            yield chunk

    async def fake_anthropic(**_: Any) -> Any:
        return fake_iter()

    body = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    for client in _client_with(
        anthropic_completion=fake_anthropic, routing=_anthropic_translate_cfg()
    ):
        with client.stream("POST", "/v1/messages", json=body) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            text = b"".join(resp.iter_bytes()).decode()

    event_types = [
        line[len("event: ") :] for line in text.splitlines() if line.startswith("event: ")
    ]
    assert event_types == ["message_start", "content_block_delta", "message_stop"]


@pytest.mark.integration
def test_count_tokens_endpoint_calls_acount_tokens() -> None:
    received: dict[str, Any] = {}

    async def fake_count(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return {"total_tokens": 9}

    body = {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": "hello there"}],
    }
    for client in _client_with(count_tokens_completion=fake_count):
        resp = client.post("/v1/messages/count_tokens", json=body)

    assert resp.status_code == 200
    assert resp.json() == {"input_tokens": 9}
    # dispatch_model gets the openai/ prefix from the test fixture's rule.
    assert received["model"] == "openai/claude-3-5-sonnet-20241022"
    assert received["messages"] == body["messages"]


@pytest.mark.integration
def test_count_tokens_endpoint_forwards_system_and_tools() -> None:
    received: dict[str, Any] = {}

    async def fake_count(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return {"total_tokens": 4}

    body = {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": "hi"}],
        "system": "Be concise.",
        "tools": [{"name": "x", "input_schema": {"type": "object"}}],
    }
    for client in _client_with(count_tokens_completion=fake_count):
        resp = client.post("/v1/messages/count_tokens", json=body)

    assert resp.status_code == 200
    assert received["system"] == "Be concise."
    assert received["tools"][0]["name"] == "x"


@pytest.mark.integration
def test_messages_forwards_inbound_headers_to_dispatch() -> None:
    """authorization, anthropic-beta, anthropic-version flow into extra_headers."""
    received: dict[str, Any] = {}

    async def fake_anthropic(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return _ANTHROPIC_MESSAGE_RESPONSE

    body = {
        "model": "claude-haiku",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    }
    for client in _client_with(
        anthropic_completion=fake_anthropic, routing=_anthropic_translate_cfg()
    ):
        resp = client.post(
            "/v1/messages",
            json=body,
            headers={
                "Authorization": "Bearer test-oauth-token",
                "anthropic-beta": "feature-x,feature-y",
                "anthropic-version": "2023-06-01",
                "x-custom-trace": "abc123",
            },
        )

    assert resp.status_code == 200
    forwarded = received["extra_headers"]
    assert forwarded["authorization"] == "Bearer test-oauth-token"
    assert forwarded["anthropic-beta"] == "feature-x,feature-y"
    assert forwarded["anthropic-version"] == "2023-06-01"
    assert forwarded["x-custom-trace"] == "abc123"
    forwarded_keys = {k.lower() for k in forwarded}
    assert "host" not in forwarded_keys
    assert "content-length" not in forwarded_keys
    assert "content-type" not in forwarded_keys


@pytest.mark.integration
def test_chat_completions_forwards_inbound_headers_to_dispatch() -> None:
    received: dict[str, Any] = {}

    async def fake_completion(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return _OPENAI_CHAT_RESPONSE

    for client in _client_with(chat_completion=fake_completion):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16,
            },
            headers={"Authorization": "Bearer key", "openai-organization": "org_123"},
        )

    assert resp.status_code == 200
    forwarded = received["extra_headers"]
    assert forwarded["authorization"] == "Bearer key"
    assert forwarded["openai-organization"] == "org_123"
    assert "content-type" not in {k.lower() for k in forwarded}


@pytest.mark.integration
def test_chat_completions_streams_sse() -> None:
    chunks = [
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4",
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4",
            "choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]

    received: dict[str, Any] = {}

    async def fake_iter() -> Any:
        for chunk in chunks:
            yield chunk

    async def fake_completion(**kwargs: Any) -> Any:
        received.update(kwargs)
        return fake_iter()

    body = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}], "stream": True}

    for client in _client_with(chat_completion=fake_completion):
        with client.stream("POST", "/v1/chat/completions", json=body) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            import json as _json  # noqa: PLC0415

            text = b"".join(resp.iter_bytes()).decode()

    assert received["stream"] is True
    events = [line[len("data: ") :] for line in text.splitlines() if line.startswith("data: ")]
    assert len(events) == 4  # 3 chunks + [DONE]
    assert events[-1] == "[DONE]"
    parsed = [_json.loads(e) for e in events[:-1]]
    assert parsed[1]["choices"][0]["delta"]["content"] == "hello"
    assert parsed[2]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.unit
def test_messages_returns_502_on_upstream_failure() -> None:
    async def boom(**_: Any) -> dict[str, Any]:
        raise RuntimeError("upstream exploded")

    body = {
        "model": "claude-haiku",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    }
    for client in _client_with(anthropic_completion=boom, routing=_anthropic_translate_cfg()):
        resp = client.post("/v1/messages", json=body)

    assert resp.status_code == 502
    assert "upstream exploded" in resp.json()["detail"]


@pytest.mark.integration
def test_responses_endpoint_translates_via_litellm() -> None:
    """/v1/responses translate-mode goes through litellm.aresponses."""
    request_body = {
        "model": "gpt-4o",
        "input": "Reply with the single word: pong",
        "max_output_tokens": 16,
    }
    response_body = {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "model": "gpt-4o",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "pong", "annotations": []}],
            }
        ],
        "status": "completed",
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }

    received: dict[str, Any] = {}

    async def fake_aresponses(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return response_body

    cfg = RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/responses"}},
                    "action": {"provider": "openai", "mode": "translate"},
                }
            ]
        }
    )
    for client in _client_with(routing=cfg, responses_completion=fake_aresponses):
        resp = client.post("/v1/responses", json=request_body)

    assert resp.status_code == 200
    assert resp.json() == response_body
    received_no_headers = {k: v for k, v in received.items() if k != "extra_headers"}
    expected = {**request_body, "model": f"openai/{request_body['model']}"}
    assert received_no_headers == expected


@pytest.mark.integration
def test_responses_endpoint_streams_sse() -> None:
    """/v1/responses streaming wraps litellm events as ``event:`` SSE frames."""
    events = [
        {"type": "response.created", "response": {"id": "resp_1", "object": "response"}},
        {"type": "response.output_text.delta", "delta": "pong"},
        {"type": "response.completed", "response": {"id": "resp_1", "status": "completed"}},
    ]

    async def fake_iter() -> Any:
        for ev in events:
            yield ev

    async def fake_aresponses(**_: Any) -> Any:
        return fake_iter()

    cfg = RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/responses"}},
                    "action": {"provider": "openai", "mode": "translate"},
                }
            ]
        }
    )
    body = {"model": "gpt-4o", "input": "hi", "stream": True}
    for client in _client_with(routing=cfg, responses_completion=fake_aresponses):
        with client.stream("POST", "/v1/responses", json=body) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            text = b"".join(resp.iter_bytes()).decode()

    event_types = [
        line[len("event: ") :] for line in text.splitlines() if line.startswith("event: ")
    ]
    assert event_types == [
        "response.created",
        "response.output_text.delta",
        "response.completed",
    ]


@pytest.mark.unit
def test_unmatched_responses_endpoint_returns_404_openai_envelope() -> None:
    cfg = RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"model": {"literal": "only-this-model"}},
                    "action": {"provider": "openai", "mode": "translate"},
                }
            ]
        }
    )
    body = {"model": "gpt-4o", "input": "x"}
    app = create_app(routing=cfg)
    with TestClient(app) as client:
        resp = client.post("/v1/responses", json=body)
    assert resp.status_code == 404
    payload = resp.json()
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["code"] == "no_route_matched"


@pytest.mark.unit
def test_unmatched_request_returns_404_with_anthropic_envelope() -> None:
    """Routing returns 404 with an Anthropic-shape error body for /v1/messages."""
    cfg = RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"model": {"literal": "only-this-model"}},
                    "action": {"provider": "openai", "mode": "translate"},
                }
            ]
        }
    )
    body = {
        "model": "gpt-4",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "x"}],
    }
    app = create_app(routing=cfg)
    with TestClient(app) as client:
        resp = client.post("/v1/messages", json=body)
    assert resp.status_code == 404
    payload = resp.json()
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "not_found_error"
    assert "gpt-4" in payload["error"]["message"]


@pytest.mark.integration
def test_responses_retrieve_passthrough_forwards_get(monkeypatch: pytest.MonkeyPatch) -> None:
    """GET /v1/responses/{id} forwards verbatim under a passthrough rule."""
    captured: dict[str, Any] = {}

    async def fake_call_passthrough(
        raw_body: bytes,
        forward_headers: dict[str, str],
        upstream_base_url: str,
        *,
        path: str,
        method: str = "POST",
        model_hint: str | None = None,
        transport: Any = None,
    ) -> tuple[int, bytes, str]:
        captured["raw_body"] = raw_body
        captured["headers"] = forward_headers
        captured["base_url"] = upstream_base_url
        captured["path"] = path
        captured["method"] = method
        return 200, b'{"id":"resp_abc","object":"response"}', "application/json"

    monkeypatch.setattr("magos.routing.dispatch.call_passthrough", fake_call_passthrough)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    cfg = RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/responses/{id}"}},
                    "action": {
                        "provider": "openai",
                        "mode": "passthrough",
                        "base_url": "https://api.openai.com",
                        "api_key_env": "OPENAI_API_KEY",
                    },
                }
            ]
        }
    )
    app = create_app(routing=cfg)
    with TestClient(app) as client:
        resp = client.get("/v1/responses/resp_abc")

    assert resp.status_code == 200
    assert resp.json() == {"id": "resp_abc", "object": "response"}
    assert captured["method"] == "GET"
    assert captured["path"] == "/v1/responses/resp_abc"
    assert captured["base_url"] == "https://api.openai.com"
    assert captured["headers"]["authorization"] == "Bearer test-key"
    assert captured["raw_body"] == b""


@pytest.mark.integration
def test_responses_cancel_passthrough_forwards_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    """DELETE /v1/responses/{id} forwards verbatim under a passthrough rule."""
    captured: dict[str, Any] = {}

    async def fake_call_passthrough(
        raw_body: bytes,
        forward_headers: dict[str, str],
        upstream_base_url: str,
        *,
        path: str,
        method: str = "POST",
        model_hint: str | None = None,
        transport: Any = None,
    ) -> tuple[int, bytes, str]:
        captured["path"] = path
        captured["method"] = method
        return 200, b'{"id":"resp_xyz","status":"cancelled"}', "application/json"

    monkeypatch.setattr("magos.routing.dispatch.call_passthrough", fake_call_passthrough)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    cfg = RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/responses/{id}"}},
                    "action": {
                        "provider": "openai",
                        "mode": "passthrough",
                        "base_url": "https://api.openai.com",
                        "api_key_env": "OPENAI_API_KEY",
                    },
                }
            ]
        }
    )
    app = create_app(routing=cfg)
    with TestClient(app) as client:
        resp = client.delete("/v1/responses/resp_xyz")

    assert resp.status_code == 200
    assert captured["method"] == "DELETE"
    assert captured["path"] == "/v1/responses/resp_xyz"


@pytest.mark.integration
def test_responses_input_items_passthrough_forwards_get(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GET /v1/responses/{id}/input_items routes via the templated endpoint."""
    captured: dict[str, Any] = {}

    async def fake_call_passthrough(
        raw_body: bytes,
        forward_headers: dict[str, str],
        upstream_base_url: str,
        *,
        path: str,
        method: str = "POST",
        model_hint: str | None = None,
        transport: Any = None,
    ) -> tuple[int, bytes, str]:
        captured["path"] = path
        captured["method"] = method
        return 200, b'{"object":"list","data":[]}', "application/json"

    monkeypatch.setattr("magos.routing.dispatch.call_passthrough", fake_call_passthrough)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    cfg = RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/responses/{id}/input_items"}},
                    "action": {
                        "provider": "openai",
                        "mode": "passthrough",
                        "base_url": "https://api.openai.com",
                        "api_key_env": "OPENAI_API_KEY",
                    },
                }
            ]
        }
    )
    app = create_app(routing=cfg)
    with TestClient(app) as client:
        resp = client.get("/v1/responses/resp_abc/input_items")

    assert resp.status_code == 200
    assert captured["method"] == "GET"
    assert captured["path"] == "/v1/responses/resp_abc/input_items"


@pytest.mark.unit
def test_responses_retrieve_unmatched_returns_404_openai_envelope() -> None:
    cfg = RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/messages"}},
                    "action": {"provider": "openai", "mode": "translate"},
                }
            ]
        }
    )
    app = create_app(routing=cfg)
    with TestClient(app) as client:
        resp = client.get("/v1/responses/resp_nope")
    assert resp.status_code == 404
    payload = resp.json()
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["code"] == "no_route_matched"


@pytest.mark.unit
def test_responses_retrieve_translate_mode_returns_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Translate mode rejects non-POST methods at dispatch time."""
    cfg = RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/responses/{id}"}},
                    "action": {"provider": "openai", "mode": "translate"},
                }
            ]
        }
    )
    app = create_app(routing=cfg)
    with TestClient(app) as client:
        resp = client.get("/v1/responses/resp_abc")
    assert resp.status_code == 503
    payload = resp.json()
    assert payload["error"]["type"] == "server_error"


@pytest.mark.unit
def test_unmatched_request_returns_404_with_openai_envelope() -> None:
    """Routing returns 404 with an OpenAI-shape error body for /v1/chat/completions."""
    cfg = RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"model": {"literal": "only-this-model"}},
                    "action": {"provider": "openai", "mode": "translate"},
                }
            ]
        }
    )
    body = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "x"}],
    }
    app = create_app(routing=cfg)
    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 404
    payload = resp.json()
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["code"] == "no_route_matched"


# --- Lifespan: Headroom pipeline warmup ---


def test_lifespan_warms_compress_pipeline_when_rule_uses_compress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If any rule has a Compress rewrite, startup must call _get_pipeline()."""
    calls: list[str] = []

    def fake_get_pipeline() -> object:
        calls.append("warmed")
        return object()

    hc = importlib.import_module("headroom.compress")
    monkeypatch.setattr(hc, "_get_pipeline", fake_get_pipeline, raising=True)

    cfg = RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/messages"}},
                    "rewrites": [{"compress": {}}],
                    "action": {"provider": "anthropic", "mode": "translate"},
                }
            ]
        }
    )
    app = create_app(routing=cfg)
    with TestClient(app):
        pass

    assert calls == ["warmed"]


def test_lifespan_skips_warmup_when_no_compress_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No Compress rewrite anywhere -> never touch headroom on startup."""
    calls: list[str] = []

    def fake_get_pipeline() -> object:
        calls.append("warmed")
        return object()

    hc = importlib.import_module("headroom.compress")
    monkeypatch.setattr(hc, "_get_pipeline", fake_get_pipeline, raising=True)

    cfg = _translate_only_cfg()
    app = create_app(routing=cfg)
    with TestClient(app):
        pass

    assert calls == []


def test_lifespan_warmup_failure_does_not_block_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken pipeline init must log + continue, not crash the app."""

    def boom() -> object:
        raise RuntimeError("pipeline init failed")

    hc = importlib.import_module("headroom.compress")
    monkeypatch.setattr(hc, "_get_pipeline", boom, raising=True)

    cfg = RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/messages"}},
                    "rewrites": [{"compress": {}}],
                    "action": {"provider": "anthropic", "mode": "translate"},
                }
            ]
        }
    )
    app = create_app(routing=cfg)
    with TestClient(app) as client:
        # App must come up despite the warmup failure; routing-layer
        # health is unaffected because compression is best-effort.
        resp = client.post("/v1/messages", json={"model": "x", "messages": []})
    # 400 (validation) or routed; either is fine — the point is "didn't crash on startup".
    assert resp.status_code != 500


# --- Lifespan: kompress_backend override ---


# Capture the real Kompress ONNX availability check at module import time,
# before any test or lifespan can replace it. The override-test pair below
# reset to this baseline at the start of each run so they're robust to
# external env state (e.g. running the suite with MAGOS_KOMPRESS_BACKEND
# already exported).
_kc_module = importlib.import_module("headroom.transforms.kompress_compressor")
_KC_ORIGINAL_IS_ONNX_AVAILABLE = _kc_module._is_onnx_available


@pytest.fixture
def _restore_kompress_onnx_check() -> Iterator[None]:
    """Restore the real ONNX-availability check around the test."""
    _kc_module._is_onnx_available = _KC_ORIGINAL_IS_ONNX_AVAILABLE  # type: ignore[attr-defined]
    try:
        yield
    finally:
        _kc_module._is_onnx_available = _KC_ORIGINAL_IS_ONNX_AVAILABLE  # type: ignore[attr-defined]


def test_lifespan_forces_pytorch_when_kompress_backend_set(
    monkeypatch: pytest.MonkeyPatch,
    _restore_kompress_onnx_check: None,
) -> None:
    """``MAGOS_KOMPRESS_BACKEND=pytorch`` flips _is_onnx_available to False
    so Headroom's loader takes the PyTorch branch on first compress call.
    """
    monkeypatch.setenv("MAGOS_KOMPRESS_BACKEND", "pytorch")
    # Pre-condition: with onnxruntime + transformers installed, this is True.
    assert _kc_module._is_onnx_available() is True

    cfg = _translate_only_cfg()
    app = create_app(routing=cfg)
    with TestClient(app):
        pass

    # After lifespan ran with backend=pytorch, the module-level binding is
    # the False-returning stub.
    assert _kc_module._is_onnx_available() is False


def test_lifespan_default_leaves_onnx_check_untouched(
    monkeypatch: pytest.MonkeyPatch,
    _restore_kompress_onnx_check: None,
) -> None:
    """Default (auto) backend must not patch the ONNX availability check."""
    monkeypatch.delenv("MAGOS_KOMPRESS_BACKEND", raising=False)

    cfg = _translate_only_cfg()
    app = create_app(routing=cfg)
    with TestClient(app):
        pass

    # Function identity preserved — no monkeypatch by lifespan.
    assert _kc_module._is_onnx_available is _KC_ORIGINAL_IS_ONNX_AVAILABLE


# ---- _maybe_inject_api_key (passthrough auth injection) ---------------------


@pytest.mark.unit
def test_inject_api_key_defaults_to_bearer_for_non_anthropic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """openai/openrouter/vultr providers get ``Authorization: Bearer`` by default."""
    from magos.routing.dispatch import _maybe_inject_api_key  # noqa: PLC0415
    from magos.routing.models import Action  # noqa: PLC0415

    monkeypatch.setenv("VULTR_API_KEY", "vk-test")
    action = Action.model_validate(
        {
            "provider": "vultr",
            "mode": "passthrough",
            "base_url": "https://api.vultrinference.com",
            "api_key_env": "VULTR_API_KEY",
        }
    )
    out = _maybe_inject_api_key({}, action)
    assert out == {"authorization": "Bearer vk-test"}
    assert "x-api-key" not in out


@pytest.mark.unit
def test_inject_api_key_anthropic_default_uses_x_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anthropic provider keeps the official ``x-api-key`` header shape."""
    from magos.routing.dispatch import _maybe_inject_api_key  # noqa: PLC0415
    from magos.routing.models import Action  # noqa: PLC0415

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    action = Action.model_validate(
        {
            "provider": "anthropic",
            "mode": "passthrough",
            "base_url": "https://api.anthropic.com",
            "api_key_env": "ANTHROPIC_API_KEY",
        }
    )
    out = _maybe_inject_api_key({}, action)
    assert out == {"x-api-key": "sk-ant-test"}
    assert "authorization" not in out


@pytest.mark.unit
def test_inject_api_key_anthropic_oauth_token_uses_bearer_plus_beta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claude-Code OAuth tokens force Bearer + ``anthropic-beta: oauth-...``.

    api.anthropic.com rejects ``sk-ant-oat...`` tokens on the ``x-api-key``
    header with 401 ``invalid x-api-key``; the only accepted shape is the
    OAuth one. The detection must override both the per-provider default
    and any explicit ``auth_header`` setting on the rule, since neither
    alternative will authenticate.
    """
    from magos.routing.dispatch import _maybe_inject_api_key  # noqa: PLC0415
    from magos.routing.models import Action  # noqa: PLC0415

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-oat01-deadbeef")

    default_shape = Action.model_validate(
        {
            "provider": "anthropic",
            "mode": "passthrough",
            "base_url": "https://api.anthropic.com",
            "api_key_env": "ANTHROPIC_API_KEY",
        }
    )
    assert _maybe_inject_api_key({}, default_shape) == {
        "authorization": "Bearer sk-ant-oat01-deadbeef",
        "anthropic-beta": "oauth-2025-04-20",
    }

    # Explicit x-api-key override is intentionally ignored for OAuth tokens.
    explicit_xapikey = Action.model_validate(
        {
            "provider": "anthropic",
            "mode": "passthrough",
            "base_url": "https://api.anthropic.com",
            "api_key_env": "ANTHROPIC_API_KEY",
            "auth_header": "x-api-key",
        }
    )
    assert _maybe_inject_api_key({}, explicit_xapikey) == {
        "authorization": "Bearer sk-ant-oat01-deadbeef",
        "anthropic-beta": "oauth-2025-04-20",
    }


@pytest.mark.unit
def test_inject_api_key_explicit_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """``action.auth_header`` overrides the per-provider default both ways."""
    from magos.routing.dispatch import _maybe_inject_api_key  # noqa: PLC0415
    from magos.routing.models import Action  # noqa: PLC0415

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    anthropic_bearer = Action.model_validate(
        {
            "provider": "anthropic",
            "mode": "passthrough",
            "base_url": "https://api.anthropic.com",
            "api_key_env": "ANTHROPIC_API_KEY",
            "auth_header": "bearer",
        }
    )
    assert _maybe_inject_api_key({}, anthropic_bearer) == {"authorization": "Bearer sk-ant-test"}

    openai_xapikey = Action.model_validate(
        {
            "provider": "openai",
            "mode": "passthrough",
            "base_url": "https://api.openai.com",
            "api_key_env": "OPENAI_API_KEY",
            "auth_header": "x-api-key",
        }
    )
    assert _maybe_inject_api_key({}, openai_xapikey) == {"x-api-key": "sk-test"}


@pytest.mark.unit
def test_inject_api_key_skips_when_inbound_auth_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Client-supplied auth always wins; injection never overwrites it."""
    from magos.routing.dispatch import _maybe_inject_api_key  # noqa: PLC0415
    from magos.routing.models import Action  # noqa: PLC0415

    monkeypatch.setenv("VULTR_API_KEY", "vk-test")
    action = Action.model_validate(
        {
            "provider": "vultr",
            "mode": "passthrough",
            "base_url": "https://api.vultrinference.com",
            "api_key_env": "VULTR_API_KEY",
        }
    )
    inbound_auth = {"authorization": "Bearer client-supplied"}
    assert _maybe_inject_api_key(inbound_auth, action) == inbound_auth

    inbound_xapikey = {"x-api-key": "client-supplied"}
    assert _maybe_inject_api_key(inbound_xapikey, action) == inbound_xapikey


@pytest.mark.unit
def test_inject_api_key_noop_in_translate_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Translate mode never injects; api_key plumbing happens via litellm kwargs."""
    from magos.routing.dispatch import _maybe_inject_api_key  # noqa: PLC0415
    from magos.routing.models import Action  # noqa: PLC0415

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    action = Action.model_validate(
        {
            "provider": "openai",
            "mode": "translate",
            "api_key_env": "OPENAI_API_KEY",
        }
    )
    assert _maybe_inject_api_key({}, action) == {}
