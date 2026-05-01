"""FastAPI endpoint tests for the magos server.

Drives the server with TestClient. Each test injects a routing config via
``create_app(routing=...)`` and overrides the completion dependency so no
real upstream is contacted. Passthrough tests build a minimal config that
forces the matched rule's mode; passthrough wire behavior itself is unit-
tested in ``test_passthrough.py``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from magos import tokens
from magos.routing import RoutingConfig
from magos.server import create_app, get_completion, get_responses_completion

FIXTURES_ROOT = Path(__file__).parent / "fixtures" / "translation"


def _load(case_dir: Path, name: str) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((case_dir / name).read_text(encoding="utf-8")))


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
                    "action": {
                        "provider": provider,
                        "mode": "translate",
                        "count_tokens_mode": "local",
                    },
                },
            ]
        }
    )


def _client_with(
    completion: Any,
    *,
    routing: RoutingConfig | None = None,
    responses_completion: Any | None = None,
) -> Iterator[TestClient]:
    cfg = routing if routing is not None else _translate_only_cfg()
    app = create_app(routing=cfg)
    app.dependency_overrides[get_completion] = lambda: completion
    if responses_completion is not None:
        app.dependency_overrides[get_responses_completion] = lambda: responses_completion
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


@pytest.mark.integration
def test_messages_endpoint_round_trip() -> None:
    case_dir = FIXTURES_ROOT / "simple_text"
    anthropic_request = _load(case_dir, "anthropic_request.json")
    expected_openai_request = _load(case_dir, "openai_request.json")
    openai_response = _load(case_dir, "openai_response.json")
    expected_anthropic_response = _load(case_dir, "anthropic_response.json")

    received: dict[str, Any] = {}

    async def fake_completion(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return openai_response

    # The fixture has a literal claude- model; route to anthropic translate
    # so the dispatch_model gets the anthropic/ prefix as before.
    cfg = RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/messages"}},
                    "action": {"provider": "anthropic", "mode": "translate"},
                }
            ]
        }
    )
    for client in _client_with(fake_completion, routing=cfg):
        resp = client.post("/v1/messages", json=anthropic_request)

    assert resp.status_code == 200
    body = resp.json()
    expected_dispatched = {
        **expected_openai_request,
        "model": f"anthropic/{expected_openai_request['model']}",
    }
    received_no_headers = {k: v for k, v in received.items() if k != "extra_headers"}
    assert received_no_headers == expected_dispatched
    assert "extra_headers" in received

    expected_no_id = {k: v for k, v in expected_anthropic_response.items() if k != "id"}
    body_no_id = {k: v for k, v in body.items() if k != "id"}
    assert body_no_id == expected_no_id


@pytest.mark.integration
def test_chat_completions_endpoint_passes_through() -> None:
    openai_request = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
    }
    openai_response = {
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

    received: dict[str, Any] = {}

    async def fake_completion(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return openai_response

    for client in _client_with(fake_completion):
        resp = client.post("/v1/chat/completions", json=openai_request)

    assert resp.status_code == 200
    assert resp.json() == openai_response
    expected_dispatched = {**openai_request, "model": f"openai/{openai_request['model']}"}
    received_no_headers = {k: v for k, v in received.items() if k != "extra_headers"}
    assert received_no_headers == expected_dispatched
    assert "extra_headers" in received


@pytest.mark.integration
def test_messages_streams_anthropic_events() -> None:
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
            "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]

    async def fake_iter() -> Any:
        for chunk in chunks:
            yield chunk

    async def fake_completion(**_: Any) -> Any:
        return fake_iter()

    body = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    cfg = RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/messages"}},
                    "action": {"provider": "anthropic", "mode": "translate"},
                }
            ]
        }
    )
    for client in _client_with(fake_completion, routing=cfg):
        with client.stream("POST", "/v1/messages", json=body) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            text = b"".join(resp.iter_bytes()).decode()

    event_types = [
        line[len("event: ") :] for line in text.splitlines() if line.startswith("event: ")
    ]
    assert event_types == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    data_lines = [line[len("data: ") :] for line in text.splitlines() if line.startswith("data: ")]
    parsed = [json.loads(line) for line in data_lines]
    assert parsed[2]["delta"] == {"type": "text_delta", "text": "hi"}
    assert parsed[4]["delta"]["stop_reason"] == "end_turn"
    assert parsed[0]["message"]["usage"]["input_tokens"] > 0


@pytest.mark.unit
def test_messages_streaming_returns_400_on_invalid_request() -> None:
    cfg = _translate_only_cfg()
    app = create_app(routing=cfg)
    body = {"model": "x", "stream": True}  # missing required fields
    with TestClient(app) as client:
        resp = client.post("/v1/messages", json=body)
    assert resp.status_code == 400


@pytest.mark.integration
def test_count_tokens_endpoint_local_path() -> None:
    """count_tokens_mode=local triggers the local estimator."""
    cfg = _translate_only_cfg()  # default for /v1/messages/count_tokens is local
    body = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hello there"}],
    }
    app = create_app(routing=cfg)
    with TestClient(app) as client:
        resp = client.post("/v1/messages/count_tokens", json=body)
    assert resp.status_code == 200
    payload = resp.json()
    assert isinstance(payload["input_tokens"], int)
    assert payload["input_tokens"] > 0


@pytest.mark.integration
def test_count_tokens_endpoint_uses_passthrough_when_rule_says_so() -> None:
    """count_tokens_mode=passthrough on an anthropic rule -> registered impl."""
    captured: dict[str, Any] = {}

    async def fake_passthrough(
        req: dict[str, Any], *, forward_headers: dict[str, str] | None = None
    ) -> int:
        captured["model"] = req["model"]
        captured["forward_headers"] = forward_headers
        return 4242

    cfg = RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/messages/count_tokens"}},
                    "action": {
                        "provider": "anthropic",
                        "mode": "passthrough",
                        "base_url": "https://api.anthropic.com",
                        "count_tokens_mode": "passthrough",
                    },
                }
            ]
        }
    )
    body = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    }
    app = create_app(routing=cfg)
    with (
        patch.dict(tokens.PASSTHROUGH_DISPATCH, {"anthropic": fake_passthrough}),
        TestClient(app) as client,
    ):
        resp = client.post("/v1/messages/count_tokens", json=body)

    assert resp.status_code == 200
    assert resp.json() == {"input_tokens": 4242}
    assert captured["model"] == "claude-3-5-sonnet-20241022"


@pytest.mark.unit
def test_count_tokens_endpoint_returns_400_on_invalid_request() -> None:
    cfg = _translate_only_cfg()
    app = create_app(routing=cfg)
    with TestClient(app) as client:
        resp = client.post("/v1/messages/count_tokens", json={"model": "x"})
    assert resp.status_code == 400


@pytest.mark.integration
def test_messages_forwards_inbound_headers_to_dispatch() -> None:
    """authorization, anthropic-beta, anthropic-version flow into extra_headers."""
    case_dir = FIXTURES_ROOT / "simple_text"
    anthropic_request = _load(case_dir, "anthropic_request.json")
    openai_response = _load(case_dir, "openai_response.json")

    received: dict[str, Any] = {}

    async def fake_completion(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return openai_response

    cfg = RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/messages"}},
                    "action": {"provider": "anthropic", "mode": "translate"},
                }
            ]
        }
    )
    for client in _client_with(fake_completion, routing=cfg):
        resp = client.post(
            "/v1/messages",
            json=anthropic_request,
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
    openai_request = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
    }
    openai_response = {
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

    received: dict[str, Any] = {}

    async def fake_completion(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return openai_response

    for client in _client_with(fake_completion):
        resp = client.post(
            "/v1/chat/completions",
            json=openai_request,
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

    for client in _client_with(fake_completion):
        with client.stream("POST", "/v1/chat/completions", json=body) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            text = b"".join(resp.iter_bytes()).decode()

    assert received["stream"] is True
    events = [line[len("data: ") :] for line in text.splitlines() if line.startswith("data: ")]
    assert len(events) == 4  # 3 chunks + [DONE]
    assert events[-1] == "[DONE]"
    parsed = [json.loads(e) for e in events[:-1]]
    assert parsed[1]["choices"][0]["delta"]["content"] == "hello"
    assert parsed[2]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.unit
def test_messages_returns_400_on_invalid_request() -> None:
    cfg = _translate_only_cfg()
    app = create_app(routing=cfg)
    with TestClient(app) as client:
        resp = client.post("/v1/messages", json={"model": "x"})  # missing required fields
    assert resp.status_code == 400


@pytest.mark.unit
def test_messages_returns_502_on_upstream_failure() -> None:
    case_dir = FIXTURES_ROOT / "simple_text"
    anthropic_request = _load(case_dir, "anthropic_request.json")

    async def boom(**_: Any) -> dict[str, Any]:
        raise RuntimeError("upstream exploded")

    cfg = RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/messages"}},
                    "action": {"provider": "anthropic", "mode": "translate"},
                }
            ]
        }
    )
    for client in _client_with(boom, routing=cfg):
        resp = client.post("/v1/messages", json=anthropic_request)

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
    for client in _client_with(
        completion=lambda **_: None,  # not used for /v1/responses
        routing=cfg,
        responses_completion=fake_aresponses,
    ):
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
    for client in _client_with(
        completion=lambda **_: None,
        routing=cfg,
        responses_completion=fake_aresponses,
    ):
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
    assert captured["headers"]["x-api-key"] == "test-key"
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
