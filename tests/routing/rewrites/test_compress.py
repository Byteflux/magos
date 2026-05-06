"""``compress`` rewrite tests: chat-shape, /v1/responses, model_limit resolution."""

from __future__ import annotations

from typing import Any

import litellm
import pytest

from magos.compression import ApplyResult, PipelineConfig
from magos.routing import Compress, CompressOptions
from magos.routing.rewrites import apply_rewrites
from magos.routing.rewrites import compress as rw
from magos.routing.rewrites.compress import token_mode as tm
from tests.routing._helpers import make_req

# --- Skip / no-op cases ---


def test_compress_skipped_on_responses_endpoint() -> None:
    req = make_req(
        endpoint="/v1/responses",
        body={"model": "x", "input": "hello"},
    )
    out = apply_rewrites(req, [Compress(compress=CompressOptions())])
    assert out is req


def test_compress_no_messages_is_noop() -> None:
    req = make_req(body={"model": "x"})
    out = apply_rewrites(req, [Compress(compress=CompressOptions())])
    assert out is req


def test_compress_empty_messages_is_noop() -> None:
    req = make_req(body={"model": "x", "messages": []})
    out = apply_rewrites(req, [Compress(compress=CompressOptions())])
    assert out is req


# --- Chat-shape token-mode pipeline ---


def test_compress_token_mode_applies_pipeline_and_marks_dirty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_apply(
        *,
        messages: list[dict[str, Any]],
        model: str,
        model_limit: int,
        config: PipelineConfig,
        provider_name: str,
        context: str | None = None,
        biases: dict[str, float] | None = None,
        **_extra: Any,
    ) -> ApplyResult:
        captured["messages"] = messages
        captured["model"] = model
        captured["model_limit"] = model_limit
        captured["config"] = config
        captured["provider_name"] = provider_name
        return ApplyResult(
            messages=[{"role": "user", "content": "shorter"}],
            tokens_before=100,
            tokens_after=60,
            tokens_saved=40,
            transforms_applied=["ContentRouter"],
        )

    monkeypatch.setattr(tm, "apply", fake_apply, raising=True)

    req = make_req(
        body={
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "verbose original"}],
        }
    )
    out = apply_rewrites(req, [Compress(compress=CompressOptions(target_ratio=0.5))])

    assert out.body_dirty is True
    assert out.body["messages"] == [{"role": "user", "content": "shorter"}]
    assert captured["model"] == "claude-sonnet-4-5"
    assert captured["provider_name"] == "anthropic"
    assert isinstance(captured["config"], PipelineConfig)
    assert isinstance(captured["model_limit"], int) and captured["model_limit"] > 0


def test_compress_token_mode_chat_endpoint_uses_openai_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_providers: list[str] = []

    def fake_apply(
        *,
        messages: list[dict[str, Any]],
        model: str,
        model_limit: int,
        config: PipelineConfig,
        provider_name: str,
        context: str | None = None,
        biases: dict[str, float] | None = None,
        **_extra: Any,
    ) -> ApplyResult:
        seen_providers.append(provider_name)
        return ApplyResult(
            messages=messages,
            tokens_before=10,
            tokens_after=10,
            tokens_saved=0,
        )

    monkeypatch.setattr(tm, "apply", fake_apply, raising=True)

    req = make_req(
        endpoint="/v1/chat/completions",
        body={"model": "gpt-4o", "messages": [{"role": "user", "content": "x"}]},
    )
    apply_rewrites(req, [Compress(compress=CompressOptions())])

    assert seen_providers == ["openai"]


def test_compress_options_forward_to_apply_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """CompressOptions runtime hints (target_ratio, protect_recent, etc.)
    must reach magos.compression.apply, not get silently dropped."""
    captured: dict[str, Any] = {}

    def fake_apply(**kwargs: Any) -> ApplyResult:
        captured.update(kwargs)
        return ApplyResult(
            messages=kwargs["messages"], tokens_before=10, tokens_after=8, tokens_saved=2
        )

    monkeypatch.setattr(tm, "apply", fake_apply, raising=True)

    req = make_req(
        body={"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "x"}]}
    )
    apply_rewrites(
        req,
        [
            Compress(
                compress=CompressOptions(
                    compress_user_messages=True,
                    protect_recent=0,
                    target_ratio=0.5,
                    min_tokens_to_compress=100,
                    kompress_model="custom/model",
                )
            )
        ],
    )

    assert captured["compress_user_messages"] is True
    assert captured["protect_recent"] == 0
    assert captured["target_ratio"] == 0.5
    assert captured["min_tokens_to_compress"] == 100
    assert captured["kompress_model"] == "custom/model"


def test_compress_token_mode_inflation_returns_request_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = [{"role": "user", "content": "input"}]

    def fake_apply(**_kwargs: Any) -> ApplyResult:
        return ApplyResult(
            messages=original,
            tokens_before=50,
            tokens_after=50,
            tokens_saved=0,
            inflation_reverted=True,
        )

    monkeypatch.setattr(tm, "apply", fake_apply, raising=True)

    req = make_req(body={"model": "x", "messages": original})
    out = apply_rewrites(req, [Compress(compress=CompressOptions())])

    assert out is req  # body_dirty must NOT flip when nothing changed


def test_compress_zero_savings_returns_input_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_apply(
        *,
        messages: list[dict[str, Any]],
        model: str,
        model_limit: int,
        config: PipelineConfig,
        provider_name: str,
        context: str | None = None,
        biases: dict[str, float] | None = None,
        **_extra: Any,
    ) -> ApplyResult:
        return ApplyResult(messages=messages, tokens_before=100, tokens_after=100, tokens_saved=0)

    monkeypatch.setattr(tm, "apply", fake_apply, raising=True)

    req = make_req(body={"model": "x", "messages": [{"role": "user", "content": "hi"}]})
    out = apply_rewrites(req, [Compress(compress=CompressOptions())])
    assert out is req


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


def test_compress_unsupported_endpoint_does_not_call_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:  # pragma: no cover
        raise AssertionError("apply() must not be called for /v1/responses")

    monkeypatch.setattr(tm, "apply", boom, raising=True)

    req = make_req(
        endpoint="/v1/responses",
        body={"model": "x", "input": "hello"},
    )
    out = apply_rewrites(req, [Compress(compress=CompressOptions())])
    assert out is req


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


def test_responses_aux_endpoints_skip_compress() -> None:
    """The /v1/responses/{id} family has no body to compress; must no-op."""
    for endpoint in ("/v1/responses/{id}", "/v1/responses/{id}/input_items"):
        req = make_req(endpoint=endpoint, body={}, raw=b"")
        out = apply_rewrites(req, [Compress(compress=CompressOptions(mode="cache"))])
        assert out is req, f"{endpoint} should no-op"


# --- model_limit resolution ---


def test_resolve_model_limit_known_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Known model id (per LiteLLM's registry) returns the real limit."""

    monkeypatch.setattr(rw.model_limit, "_MODEL_LIMIT_CACHE", {})
    # gpt-4o has been in LiteLLM's registry stably; if this assert ever
    # breaks it's because LiteLLM dropped the model, not magos.
    assert rw._resolve_model_limit("gpt-4o") == 128_000


def test_resolve_model_limit_unknown_model_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rw.model_limit, "_MODEL_LIMIT_CACHE", {})
    assert rw._resolve_model_limit("totally-made-up-model-zzz") == rw._DEFAULT_MODEL_LIMIT


def test_resolve_model_limit_caches_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeat lookups for the same model don't re-call litellm."""

    cache: dict[str, int] = {}
    monkeypatch.setattr(rw.model_limit, "_MODEL_LIMIT_CACHE", cache)

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

    def fake_apply(
        *,
        messages: list[dict[str, Any]],
        model: str,
        model_limit: int,
        config: PipelineConfig,
        provider_name: str,
        context: str | None = None,
        biases: dict[str, float] | None = None,
        **_extra: Any,
    ) -> ApplyResult:
        captured["model_limit"] = model_limit
        return ApplyResult(
            messages=[{"role": "user", "content": "x"}],
            tokens_before=10,
            tokens_after=10,
            tokens_saved=0,
        )

    monkeypatch.setattr(tm, "apply", fake_apply, raising=True)

    # Spy on _resolve_model_limit to assert it's NOT called when explicit.

    called: list[str] = []

    def spy_resolve(_model: str, default: int = 0) -> int:
        called.append(_model)
        return 999_999  # should not be used

    monkeypatch.setattr(rw.token_mode, "_resolve_model_limit", spy_resolve)

    req = make_req(
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

    def fake_apply(
        *,
        messages: list[dict[str, Any]],
        model: str,
        model_limit: int,
        config: PipelineConfig,
        provider_name: str,
        context: str | None = None,
        biases: dict[str, float] | None = None,
        **_extra: Any,
    ) -> ApplyResult:
        captured["model_limit"] = model_limit
        return ApplyResult(
            messages=[{"role": "user", "content": "x"}],
            tokens_before=10,
            tokens_after=10,
            tokens_saved=0,
        )

    monkeypatch.setattr(tm, "apply", fake_apply, raising=True)

    monkeypatch.setattr(rw.model_limit, "_MODEL_LIMIT_CACHE", {"gpt-4o": 128_000})

    req = make_req(body={"model": "gpt-4o", "messages": [{"role": "user", "content": "verbose"}]})
    apply_rewrites(req, [Compress(compress=CompressOptions())])
    assert captured["model_limit"] == 128_000


# --- Prefix-cache tracker integration ---


def test_compress_token_mode_passes_frozen_count_from_tracker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First-turn cold-start: tracker reports frozen_count=0."""
    from magos.cache import store as store_mod  # noqa: PLC0415

    captured: dict[str, Any] = {}

    def fake_apply(**kwargs: Any) -> ApplyResult:
        captured.update(kwargs)
        return ApplyResult(
            messages=kwargs["messages"], tokens_before=10, tokens_after=8, tokens_saved=2
        )

    monkeypatch.setattr(tm, "apply", fake_apply, raising=True)
    monkeypatch.setattr(store_mod, "_STORE", store_mod.TrackerStore())

    req = make_req(
        body={"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "x"}]}
    )
    apply_rewrites(req, [Compress(compress=CompressOptions())])

    assert captured["frozen_message_count"] == 0


def test_compress_token_mode_registers_post_response_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The compress rewrite appends exactly one hook to req.post_response_hooks."""
    from magos.cache import store as store_mod  # noqa: PLC0415

    def fake_apply(**kwargs: Any) -> ApplyResult:
        return ApplyResult(
            messages=kwargs["messages"], tokens_before=100, tokens_after=60, tokens_saved=40
        )

    monkeypatch.setattr(tm, "apply", fake_apply, raising=True)
    monkeypatch.setattr(store_mod, "_STORE", store_mod.TrackerStore())

    req = make_req(
        body={
            "model": "claude-sonnet-4-5",
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "x"}],
        }
    )
    out = apply_rewrites(req, [Compress(compress=CompressOptions())])

    assert len(out.post_response_hooks) == 1


def test_compress_token_mode_hook_updates_tracker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Firing the registered hook reaches PrefixCacheTracker.update_from_response."""
    from magos.cache import PrefixCacheTracker, store as store_mod  # noqa: PLC0415, I001
    from magos.egress.usage import Usage  # noqa: PLC0415

    update_calls: list[dict[str, Any]] = []

    def fake_update(
        self: PrefixCacheTracker,
        cache_read_tokens: int,
        cache_write_tokens: int,
        messages: list[dict[str, Any]],
        message_token_counts: list[int] | None = None,
        original_messages: list[dict[str, Any]] | None = None,
    ) -> None:
        update_calls.append(
            {
                "cache_read": cache_read_tokens,
                "cache_write": cache_write_tokens,
                "n_messages": len(messages),
            }
        )

    monkeypatch.setattr(store_mod, "_STORE", store_mod.TrackerStore())
    monkeypatch.setattr(PrefixCacheTracker, "update_from_response", fake_update)

    def fake_apply(**kwargs: Any) -> ApplyResult:
        return ApplyResult(
            messages=kwargs["messages"], tokens_before=100, tokens_after=60, tokens_saved=40
        )

    monkeypatch.setattr(tm, "apply", fake_apply, raising=True)

    req = make_req(
        body={"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "x"}]}
    )
    out = apply_rewrites(req, [Compress(compress=CompressOptions())])

    # Now fire the registered hook with a fake Usage.
    out.post_response_hooks[0](Usage(input=200, output=100, cache_read=4000, cache_write=0))

    assert len(update_calls) == 1
    assert update_calls[0]["cache_read"] == 4000
    assert update_calls[0]["cache_write"] == 0


def test_compress_token_mode_no_hook_when_inflation_reverted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If apply() returns inflation_reverted=True, no rewrite happens. The hook
    still registers though, because the upstream cache state still needs to be
    tracked to inform the next turn."""
    from magos.cache import store as store_mod  # noqa: PLC0415

    def fake_apply(**kwargs: Any) -> ApplyResult:
        return ApplyResult(
            messages=kwargs["messages"],
            tokens_before=50,
            tokens_after=50,
            tokens_saved=0,
            inflation_reverted=True,
        )

    monkeypatch.setattr(tm, "apply", fake_apply, raising=True)
    monkeypatch.setattr(store_mod, "_STORE", store_mod.TrackerStore())

    req = make_req(
        body={"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "x"}]}
    )
    out = apply_rewrites(req, [Compress(compress=CompressOptions())])

    # body unchanged, but hook still registered for tracker observability.
    assert out.body == req.body
    assert len(out.post_response_hooks) == 1
