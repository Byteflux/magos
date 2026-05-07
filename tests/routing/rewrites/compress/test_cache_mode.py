"""Cache-mode compression tests: chat-shape aligner + /v1/responses Phase 1."""

from __future__ import annotations

from typing import Any

import pytest

from magos.compression import ApplyResult
from magos.compression.engine import token as tm
from magos.routing import Compress, CompressOptions
from magos.routing.rewrites import apply_rewrites
from tests.routing._helpers import make_req

# --- Chat-shape cache mode ---


def test_compress_cache_mode_runs_aligner_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """``mode: cache`` must not invoke the full compress() pipeline."""

    def boom(*args: Any, **kwargs: Any) -> ApplyResult:  # pragma: no cover
        raise AssertionError("apply() must not be called in cache mode")

    monkeypatch.setattr(tm, "apply", boom, raising=True)

    # The DynamicContentDetector (Tier 1 regex) extracts UUIDs from the
    # static prefix into a dynamic-context tail. UUID detection is the
    # detector's value-add over the legacy date-only regex path.
    uuid = "550e8400-e29b-41d4-a716-446655440000"
    req = make_req(
        body={
            "model": "claude-sonnet-4-5",
            "messages": [
                {"role": "system", "content": f"Session: {uuid}. Be concise."},
                {"role": "user", "content": "hello"},
            ],
        }
    )
    out = apply_rewrites(req, [Compress(compress=CompressOptions(mode="cache"))])
    assert out.body_dirty is True
    sys_content = out.body["messages"][0]["content"]
    static_prefix = sys_content.split("[Dynamic Context]")[0]
    assert uuid not in static_prefix
    assert uuid in sys_content


# --- /v1/responses Phase 1 (instructions cache alignment) ---


def test_responses_cache_mode_aligns_instructions() -> None:
    """``mode: cache`` extracts dynamic content from the Responses
    ``instructions`` field and writes the stabilised string back."""
    uuid = "550e8400-e29b-41d4-a716-446655440000"
    req = make_req(
        endpoint="/v1/responses",
        body={
            "model": "gpt-4o",
            "instructions": f"Session: {uuid}. Be concise.",
            "input": "hello",
        },
    )
    out = apply_rewrites(req, [Compress(compress=CompressOptions(mode="cache"))])

    assert out.body_dirty is True
    new_instructions = out.body["instructions"]
    static_prefix = new_instructions.split("[Dynamic Context]")[0]
    assert uuid not in static_prefix
    assert uuid in new_instructions
    # Other Responses fields preserved verbatim.
    assert out.body["input"] == "hello"
    assert out.body["model"] == "gpt-4o"


def test_responses_cache_mode_noop_when_no_dynamic_content() -> None:
    """Static instructions string -> aligner declares no-op, body unchanged."""
    req = make_req(
        endpoint="/v1/responses",
        body={
            "model": "gpt-4o",
            "instructions": "You are a helpful assistant. Be concise.",
            "input": "hello",
        },
    )
    out = apply_rewrites(req, [Compress(compress=CompressOptions(mode="cache"))])
    assert out is req


def test_responses_cache_mode_noop_when_instructions_missing() -> None:
    req = make_req(
        endpoint="/v1/responses",
        body={"model": "gpt-4o", "input": "hello"},
    )
    out = apply_rewrites(req, [Compress(compress=CompressOptions(mode="cache"))])
    assert out is req


def test_responses_cache_mode_noop_when_instructions_empty() -> None:
    req = make_req(
        endpoint="/v1/responses",
        body={"model": "gpt-4o", "instructions": "   ", "input": "hello"},
    )
    out = apply_rewrites(req, [Compress(compress=CompressOptions(mode="cache"))])
    assert out is req


def test_responses_token_mode_does_not_call_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``mode: token`` is unsupported on /v1/responses: must not call apply()."""

    def boom(*args: Any, **kwargs: Any) -> None:  # pragma: no cover
        raise AssertionError("apply() must not be called for Responses token mode")

    monkeypatch.setattr(tm, "apply", boom, raising=True)

    req = make_req(
        endpoint="/v1/responses",
        body={
            "model": "gpt-4o",
            "instructions": "Current date: 2026-05-01. Be concise.",
            "input": "hello",
        },
    )
    out = apply_rewrites(req, [Compress(compress=CompressOptions(mode="token"))])
    assert out is req
