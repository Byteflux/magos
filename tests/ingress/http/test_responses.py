"""``/v1/responses`` endpoint family tests.

Covers the translate-mode POST handler plus the three passthrough
auxiliary endpoints (retrieve / cancel / input_items) and their error
envelopes.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from magos.ingress.http import create_app
from magos.routing import RoutingConfig

from ._helpers import client_with


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
                    "target": {"provider": "openai", "gateway": "translate"},
                }
            ]
        }
    )
    for client in client_with(routing=cfg, responses_completion=fake_aresponses):
        resp = client.post("/v1/responses", json=request_body)

    assert resp.status_code == 200
    expected_response = {**response_body, "model": f"openai/{request_body['model']}"}
    assert resp.json() == expected_response
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
                    "target": {"provider": "openai", "gateway": "translate"},
                }
            ]
        }
    )
    body = {"model": "gpt-4o", "input": "hi", "stream": True}
    for client in client_with(routing=cfg, responses_completion=fake_aresponses):
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
                    "target": {"provider": "openai", "gateway": "translate"},
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

    monkeypatch.setattr("magos.egress.dispatch.call_passthrough", fake_call_passthrough)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    cfg = RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/responses/{id}"}},
                    "target": {
                        "provider": "openai",
                        "gateway": "passthrough",
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

    monkeypatch.setattr("magos.egress.dispatch.call_passthrough", fake_call_passthrough)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    cfg = RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/responses/{id}"}},
                    "target": {
                        "provider": "openai",
                        "gateway": "passthrough",
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

    monkeypatch.setattr("magos.egress.dispatch.call_passthrough", fake_call_passthrough)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    cfg = RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/responses/{id}/input_items"}},
                    "target": {
                        "provider": "openai",
                        "gateway": "passthrough",
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
                    "target": {"provider": "openai", "gateway": "translate"},
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
                    "target": {"provider": "openai", "gateway": "translate"},
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
