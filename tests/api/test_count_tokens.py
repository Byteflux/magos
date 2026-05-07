"""``/v1/messages/count_tokens`` endpoint tests."""

from __future__ import annotations

from typing import Any

import pytest

from tests.api._helpers import client_with


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
    for client in client_with(count_tokens_completion=fake_count):
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
    for client in client_with(count_tokens_completion=fake_count):
        resp = client.post("/v1/messages/count_tokens", json=body)

    assert resp.status_code == 200
    assert received["system"] == "Be concise."
    assert received["tools"][0]["name"] == "x"
