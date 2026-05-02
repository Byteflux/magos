"""Tests for the pure rewrite applicator."""

from __future__ import annotations

from typing import Any

import headroom
import litellm
import pytest

from magos.routing import (
    AddHeader,
    Compress,
    CompressOptions,
    JqPatch,
    NamedValue,
    RemoveHeader,
    RoutedRequest,
    SetHeader,
    SetModel,
)
from magos.routing import rewrites as rw
from magos.routing.request import Endpoint
from magos.routing.rewrites import RewriteError, apply_rewrites


def _req(
    *,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    raw: bytes = b"",
    body_dirty: bool = False,
    endpoint: Endpoint = "/v1/messages",
) -> RoutedRequest:
    return RoutedRequest(
        endpoint=endpoint,
        headers=headers or {},
        body=body or {},
        raw_body=raw,
        body_dirty=body_dirty,
    )


# --- Identity / no-op ---


def test_empty_rewrite_list_returns_input_identity() -> None:
    req = _req(body={"model": "x"})
    assert apply_rewrites(req, []) is req


# --- SetModel ---


def test_set_model_replaces_value_and_marks_dirty() -> None:
    req = _req(body={"model": "old", "max_tokens": 8})
    out = apply_rewrites(req, [SetModel(set_model="new")])
    assert out.body["model"] == "new"
    assert out.body["max_tokens"] == 8
    assert out.body_dirty is True


def test_set_model_does_not_mutate_input() -> None:
    body = {"model": "old"}
    req = _req(body=body)
    apply_rewrites(req, [SetModel(set_model="new")])
    assert body == {"model": "old"}


# --- SetHeader ---


def test_set_header_inserts() -> None:
    req = _req()
    out = apply_rewrites(req, [SetHeader(set_header=NamedValue(name="x-foo", value="bar"))])
    assert out.headers["x-foo"] == "bar"
    assert out.body_dirty is False


def test_set_header_overwrites() -> None:
    req = _req(headers={"x-foo": "old"})
    out = apply_rewrites(req, [SetHeader(set_header=NamedValue(name="x-foo", value="new"))])
    assert out.headers["x-foo"] == "new"


def test_set_header_lowercases_name() -> None:
    req = _req(headers={"x-foo": "old"})
    out = apply_rewrites(req, [SetHeader(set_header=NamedValue(name="X-Foo", value="new"))])
    assert out.headers == {"x-foo": "new"}


def test_set_header_idempotent_repeat() -> None:
    req = _req()
    once = apply_rewrites(req, [SetHeader(set_header=NamedValue(name="x-foo", value="bar"))])
    twice = apply_rewrites(
        req,
        [
            SetHeader(set_header=NamedValue(name="x-foo", value="other")),
            SetHeader(set_header=NamedValue(name="x-foo", value="bar")),
        ],
    )
    assert once.headers == twice.headers


# --- AddHeader ---


def test_add_header_inserts_when_absent() -> None:
    req = _req()
    out = apply_rewrites(req, [AddHeader(add_header=NamedValue(name="x-foo", value="bar"))])
    assert out.headers["x-foo"] == "bar"


def test_add_header_no_op_when_present() -> None:
    req = _req(headers={"x-foo": "existing"})
    out = apply_rewrites(req, [AddHeader(add_header=NamedValue(name="x-foo", value="new"))])
    assert out.headers["x-foo"] == "existing"


def test_add_header_collision_is_case_insensitive() -> None:
    req = _req(headers={"x-foo": "existing"})
    out = apply_rewrites(req, [AddHeader(add_header=NamedValue(name="X-Foo", value="new"))])
    assert out.headers == {"x-foo": "existing"}


# --- RemoveHeader ---


def test_remove_header_drops_present() -> None:
    req = _req(headers={"x-foo": "bar", "x-keep": "1"})
    out = apply_rewrites(req, [RemoveHeader(remove_header="x-foo")])
    assert "x-foo" not in out.headers
    assert out.headers["x-keep"] == "1"


def test_remove_header_missing_is_noop() -> None:
    req = _req(headers={"x-keep": "1"})
    out = apply_rewrites(req, [RemoveHeader(remove_header="x-foo")])
    assert out.headers == {"x-keep": "1"}


def test_remove_header_lowercases_name() -> None:
    req = _req(headers={"x-foo": "bar"})
    out = apply_rewrites(req, [RemoveHeader(remove_header="X-Foo")])
    assert "x-foo" not in out.headers


# --- JqPatch ---


def test_jq_patch_adds_field_and_marks_dirty() -> None:
    req = _req(body={"model": "x"})
    out = apply_rewrites(req, [JqPatch(jq_patch=".max_tokens //= 1024")])
    assert out.body["max_tokens"] == 1024
    assert out.body["model"] == "x"
    assert out.body_dirty is True


def test_jq_patch_overwrites_field() -> None:
    req = _req(body={"model": "old"})
    out = apply_rewrites(req, [JqPatch(jq_patch='.model = "new"')])
    assert out.body["model"] == "new"


def test_jq_patch_non_object_result_raises() -> None:
    req = _req(body={"model": "x"})
    with pytest.raises(RewriteError, match="must be a JSON object"):
        apply_rewrites(req, [JqPatch(jq_patch=".model")])


def test_jq_patch_null_result_raises() -> None:
    req = _req(body={})
    with pytest.raises(RewriteError, match="NoneType"):
        apply_rewrites(req, [JqPatch(jq_patch=".missing")])


# --- Header-only ops do not flip body_dirty ---


def test_header_only_ops_preserve_body_dirty_false() -> None:
    req = _req(headers={"x-old": "1"})
    out = apply_rewrites(
        req,
        [
            SetHeader(set_header=NamedValue(name="x-new", value="2")),
            RemoveHeader(remove_header="x-old"),
            AddHeader(add_header=NamedValue(name="x-other", value="3")),
        ],
    )
    assert out.body_dirty is False


def test_body_dirty_persists_once_set() -> None:
    req = _req(body={"model": "x"})
    out = apply_rewrites(
        req,
        [
            SetModel(set_model="y"),
            SetHeader(set_header=NamedValue(name="x-foo", value="bar")),
        ],
    )
    assert out.body_dirty is True


# --- Sequencing ---


def test_sequential_rewrites_apply_in_order() -> None:
    req = _req(body={"model": "a"})
    out = apply_rewrites(
        req,
        [
            SetModel(set_model="b"),
            JqPatch(jq_patch='.model = "c"'),
        ],
    )
    assert out.body["model"] == "c"


def test_input_request_is_not_mutated() -> None:
    headers = {"x-foo": "bar"}
    body = {"model": "old", "max_tokens": 8}
    req = _req(headers=headers, body=body)
    apply_rewrites(
        req,
        [
            SetModel(set_model="new"),
            SetHeader(set_header=NamedValue(name="x-foo", value="overwritten")),
            RemoveHeader(remove_header="x-foo"),
            JqPatch(jq_patch=".max_tokens = 16"),
        ],
    )
    assert headers == {"x-foo": "bar"}
    assert body == {"model": "old", "max_tokens": 8}


# --- Compress ---


def test_compress_skipped_on_responses_endpoint() -> None:
    req = _req(
        endpoint="/v1/responses",
        body={"model": "x", "input": "hello"},
    )
    out = apply_rewrites(req, [Compress(compress=CompressOptions())])
    assert out is req


def test_compress_no_messages_is_noop() -> None:
    req = _req(body={"model": "x"})
    out = apply_rewrites(req, [Compress(compress=CompressOptions())])
    assert out is req


def test_compress_empty_messages_is_noop() -> None:
    req = _req(body={"model": "x", "messages": []})
    out = apply_rewrites(req, [Compress(compress=CompressOptions())])
    assert out is req


class _StubResult:
    def __init__(
        self,
        messages: list[dict[str, Any]],
        *,
        before: int = 100,
        after: int = 60,
        transforms: list[str] | None = None,
    ) -> None:
        self.messages = messages
        self.tokens_before = before
        self.tokens_after = after
        self.tokens_saved = before - after
        self.compression_ratio = (before - after) / before if before > 0 else 0.0
        self.transforms_applied = transforms or ["stub"]


def test_compress_token_mode_applies_pipeline_and_marks_dirty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_compress(messages: list[dict[str, Any]], **kwargs: Any) -> _StubResult:
        captured["messages"] = messages
        captured["model"] = kwargs.get("model")
        captured["model_limit"] = kwargs.get("model_limit")
        captured["config"] = kwargs.get("config")
        return _StubResult([{"role": "user", "content": "shorter"}])

    monkeypatch.setattr(headroom, "compress", fake_compress, raising=True)

    req = _req(
        body={
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "verbose original"}],
        }
    )
    out = apply_rewrites(req, [Compress(compress=CompressOptions(target_ratio=0.5))])

    assert out.body_dirty is True
    assert out.body["messages"] == [{"role": "user", "content": "shorter"}]
    assert captured["model"] == "claude-sonnet-4-5"
    assert captured["config"].target_ratio == 0.5
    # model_limit is plumbed through (auto-resolved or default fallback).
    assert isinstance(captured["model_limit"], int)
    assert captured["model_limit"] > 0


def test_compress_zero_savings_returns_input_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_compress(messages: list[dict[str, Any]], **kwargs: Any) -> _StubResult:
        return _StubResult(messages, before=100, after=100)

    monkeypatch.setattr(headroom, "compress", fake_compress, raising=True)

    req = _req(body={"model": "x", "messages": [{"role": "user", "content": "hi"}]})
    out = apply_rewrites(req, [Compress(compress=CompressOptions())])
    assert out is req


def test_compress_cache_mode_runs_aligner_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """``mode: cache`` must not invoke the full compress() pipeline."""

    def fake_compress(*args: Any, **kwargs: Any) -> _StubResult:  # pragma: no cover
        raise AssertionError("compress() must not be called in cache mode")

    monkeypatch.setattr(headroom, "compress", fake_compress, raising=True)

    # The DynamicContentDetector (Tier 1 regex) extracts UUIDs from the
    # static prefix into a dynamic-context tail. UUID detection is the
    # detector's value-add over the legacy date-only regex path.
    uuid = "550e8400-e29b-41d4-a716-446655440000"
    req = _req(
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


def test_compress_unsupported_endpoint_does_not_call_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:  # pragma: no cover
        raise AssertionError("compress() must not be called for /v1/responses")

    monkeypatch.setattr(headroom, "compress", boom, raising=True)

    req = _req(
        endpoint="/v1/responses",
        body={"model": "x", "input": "hello"},
    )
    out = apply_rewrites(req, [Compress(compress=CompressOptions())])
    assert out is req


# --- Compress: /v1/responses Phase 1 (instructions cache alignment) ---


def test_responses_cache_mode_aligns_instructions() -> None:
    """``mode: cache`` extracts dynamic content from the Responses
    ``instructions`` field and writes the stabilised string back."""
    uuid = "550e8400-e29b-41d4-a716-446655440000"
    req = _req(
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
    req = _req(
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
    req = _req(
        endpoint="/v1/responses",
        body={"model": "gpt-4o", "input": "hello"},
    )
    out = apply_rewrites(req, [Compress(compress=CompressOptions(mode="cache"))])
    assert out is req


def test_responses_cache_mode_noop_when_instructions_empty() -> None:
    req = _req(
        endpoint="/v1/responses",
        body={"model": "gpt-4o", "instructions": "   ", "input": "hello"},
    )
    out = apply_rewrites(req, [Compress(compress=CompressOptions(mode="cache"))])
    assert out is req


def test_responses_token_mode_does_not_call_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``mode: token`` is unsupported on /v1/responses — must not call compress()."""

    def boom(*args: Any, **kwargs: Any) -> None:  # pragma: no cover
        raise AssertionError("compress() must not be called for Responses token mode")

    monkeypatch.setattr(headroom, "compress", boom, raising=True)

    req = _req(
        endpoint="/v1/responses",
        body={
            "model": "gpt-4o",
            "instructions": "Current date: 2026-05-01. Be concise.",
            "input": "hello",
        },
    )
    out = apply_rewrites(req, [Compress(compress=CompressOptions(mode="token"))])
    assert out is req


def test_responses_aux_endpoints_skip_compress() -> None:
    """The /v1/responses/{id} family has no body to compress; must no-op."""
    for endpoint in ("/v1/responses/{id}", "/v1/responses/{id}/input_items"):
        req = _req(endpoint=endpoint, body={}, raw=b"")
        out = apply_rewrites(req, [Compress(compress=CompressOptions(mode="cache"))])
        assert out is req, f"{endpoint} should no-op"


# --- Compress: model_limit resolution ---


def test_resolve_model_limit_known_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Known model id (per LiteLLM's registry) returns the real limit."""

    monkeypatch.setattr(rw, "_MODEL_LIMIT_CACHE", {})
    # gpt-4o has been in LiteLLM's registry stably; if this assert ever
    # breaks it's because LiteLLM dropped the model, not magos.
    assert rw._resolve_model_limit("gpt-4o") == 128_000


def test_resolve_model_limit_unknown_model_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    monkeypatch.setattr(rw, "_MODEL_LIMIT_CACHE", {})
    assert rw._resolve_model_limit("totally-made-up-model-zzz") == rw._DEFAULT_MODEL_LIMIT


def test_resolve_model_limit_caches_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeat lookups for the same model don't re-call litellm."""

    cache: dict[str, int] = {}
    monkeypatch.setattr(rw, "_MODEL_LIMIT_CACHE", cache)

    calls: list[str] = []

    def spy(model: str) -> dict[str, int]:
        calls.append(model)
        return {"max_input_tokens": 42_000}

    monkeypatch.setattr(litellm, "get_model_info", spy)

    assert rw._resolve_model_limit("foo") == 42_000
    assert rw._resolve_model_limit("foo") == 42_000
    assert calls == ["foo"]
    assert cache["foo"] == 42_000


def test_compress_uses_explicit_model_limit_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``opts.model_limit`` bypasses the LiteLLM lookup."""
    captured: dict[str, Any] = {}

    def fake_compress(messages: list[dict[str, Any]], **kwargs: Any) -> _StubResult:
        captured["model_limit"] = kwargs.get("model_limit")
        return _StubResult([{"role": "user", "content": "x"}])

    monkeypatch.setattr(headroom, "compress", fake_compress, raising=True)

    # Spy on _resolve_model_limit to assert it's NOT called when explicit.

    called: list[str] = []

    def spy_resolve(_model: str, default: int = 0) -> int:
        called.append(_model)
        return 999_999  # should not be used

    monkeypatch.setattr(rw, "_resolve_model_limit", spy_resolve)

    req = _req(
        body={
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "verbose"}],
        }
    )
    apply_rewrites(req, [Compress(compress=CompressOptions(model_limit=50_000))])
    assert captured["model_limit"] == 50_000
    assert called == [], "explicit model_limit must bypass _resolve_model_limit"


def test_compress_model_limit_auto_detect_per_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto-detected limit for the dispatch model is plumbed to compress()."""
    captured: dict[str, Any] = {}

    def fake_compress(messages: list[dict[str, Any]], **kwargs: Any) -> _StubResult:
        captured["model_limit"] = kwargs.get("model_limit")
        return _StubResult([{"role": "user", "content": "x"}])

    monkeypatch.setattr(headroom, "compress", fake_compress, raising=True)

    monkeypatch.setattr(rw, "_MODEL_LIMIT_CACHE", {"gpt-4o": 128_000})

    req = _req(body={"model": "gpt-4o", "messages": [{"role": "user", "content": "verbose"}]})
    apply_rewrites(req, [Compress(compress=CompressOptions())])
    assert captured["model_limit"] == 128_000
