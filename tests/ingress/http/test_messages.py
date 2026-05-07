"""``/v1/messages`` endpoint tests."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

from magos.ingress.http import create_app
from magos.ingress.http.handlers import get_anthropic_messages_completion
from magos.registry.schema import RegistryYaml
from magos.registry.state import ModelEntry, RegistryState
from magos.registry.store import save as save_state
from magos.routing import RoutingConfig

from ._helpers import (
    ANTHROPIC_MESSAGE_RESPONSE,
    anthropic_translate_cfg,
    client_with,
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
        return ANTHROPIC_MESSAGE_RESPONSE

    body = {
        "model": "claude-haiku",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    }
    for client in client_with(
        anthropic_completion=fake_anthropic, routing=anthropic_translate_cfg()
    ):
        resp = client.post("/v1/messages", json=body)

    assert resp.status_code == 200
    # Response model gets provider prefix added when request model lacks one
    expected_response = {**ANTHROPIC_MESSAGE_RESPONSE, "model": f"anthropic/{body['model']}"}
    assert resp.json() == expected_response
    # dispatch_model gets the anthropic/ prefix; max_tokens + messages survive.
    assert received["model"] == "anthropic/claude-haiku"
    assert received["max_tokens"] == 16
    assert received["messages"] == body["messages"]


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
    for client in client_with(
        anthropic_completion=fake_anthropic, routing=anthropic_translate_cfg()
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
def test_messages_translates_namespaced_id_to_litellm_id_via_registry(
    tmp_path: Any,
) -> None:
    """``set_model: vultr/Qwen/...`` style body model is translated to the
    registry entry's ``litellm_id`` (``custom_openai/Qwen/...``) before
    dispatch; without this LiteLLM rejects the unknown ``vultr/`` prefix.

    Also exercises the bare-id form (``Qwen/...`` with ``provider: vultr``)
    which is resolved by prepending the action's provider for lookup.
    """
    received: dict[str, Any] = {}

    async def fake_anthropic(**kwargs: Any) -> dict[str, Any]:
        received["model"] = kwargs["model"]
        return ANTHROPIC_MESSAGE_RESPONSE

    entry = ModelEntry(
        provider="vultr",
        raw_id="Qwen/Qwen3.5-397B-A17B-FP8",
        litellm_id="custom_openai/Qwen/Qwen3.5-397B-A17B-FP8",
    )
    models_path = tmp_path / "models.json"
    save_state(
        RegistryState(
            entries={entry.namespaced_id: entry},
            refreshed_at={"vultr": datetime(2026, 5, 4, tzinfo=UTC)},
        ),
        models_path,
    )
    registry_cfg = RegistryYaml.model_validate(
        {
            "providers": {"vultr": {"discovery": "noop"}},
            "registry": {"models_path": str(models_path)},
        }
    )
    routing = RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/messages"}},
                    "target": {"provider": "vultr", "gateway": "translate"},
                }
            ]
        }
    )

    app = create_app(routing=routing, registry=registry_cfg)
    app.dependency_overrides[get_anthropic_messages_completion] = lambda: fake_anthropic
    try:
        with TestClient(app) as client:
            for body_model, expected in (
                ("vultr/Qwen/Qwen3.5-397B-A17B-FP8", "custom_openai/Qwen/Qwen3.5-397B-A17B-FP8"),
                ("Qwen/Qwen3.5-397B-A17B-FP8", "custom_openai/Qwen/Qwen3.5-397B-A17B-FP8"),
            ):
                received.clear()
                resp = client.post(
                    "/v1/messages",
                    json={
                        "model": body_model,
                        "max_tokens": 8,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
                assert resp.status_code == 200, body_model
                assert received["model"] == expected, body_model
    finally:
        app.dependency_overrides.clear()


@pytest.mark.integration
def test_messages_forwards_inbound_headers_to_dispatch() -> None:
    """authorization, anthropic-beta, anthropic-version flow into extra_headers."""
    received: dict[str, Any] = {}

    async def fake_anthropic(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return ANTHROPIC_MESSAGE_RESPONSE

    body = {
        "model": "claude-haiku",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    }
    for client in client_with(
        anthropic_completion=fake_anthropic, routing=anthropic_translate_cfg()
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


@pytest.mark.unit
def test_messages_returns_502_on_upstream_failure() -> None:
    async def boom(**_: Any) -> dict[str, Any]:
        raise RuntimeError("upstream exploded")

    body = {
        "model": "claude-haiku",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    }
    for client in client_with(anthropic_completion=boom, routing=anthropic_translate_cfg()):
        resp = client.post("/v1/messages", json=body)

    assert resp.status_code == 502
    assert "upstream exploded" in resp.json()["detail"]


@pytest.mark.unit
def test_unmatched_request_returns_404_with_anthropic_envelope() -> None:
    """Routing returns 404 with an Anthropic-shape error body for /v1/messages."""
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
