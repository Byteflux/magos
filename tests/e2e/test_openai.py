"""End-to-end smoke tests for OpenAI-shape endpoints (Chat + Responses).

See ``tests/e2e/conftest.py`` for the ``MAGOS_E2E=1`` skip gate.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from magos.api import build_api

from ._helpers import MODEL, PROMPT


def test_openai_non_streaming_real() -> None:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 16,
    }
    with TestClient(build_api()) as client:
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
        TestClient(build_api()) as client,
        client.stream("POST", "/v1/chat/completions", json=body) as resp,
    ):
        assert resp.status_code == 200
        text = b"".join(resp.iter_bytes()).decode()
    assert "data: [DONE]" in text


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
    with TestClient(build_api()) as client:
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
