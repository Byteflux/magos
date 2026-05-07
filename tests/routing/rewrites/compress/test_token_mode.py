"""Token-mode compression tests + prefix-cache tracker integration."""

from __future__ import annotations

from typing import Any

import pytest

from magos.compression import ApplyResult, PipelineConfig
from magos.routing import Compress, CompressOptions
from magos.routing.rewrites import apply_rewrites
from magos.routing.rewrites.compress import token_mode as tm
from tests.routing._helpers import make_req

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
