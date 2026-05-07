"""``model_limit`` resolution: registry/litellm lookup + cache + override."""

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
