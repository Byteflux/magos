"""`apply` wraps `pipeline.apply` with an inflation guard."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from magos.compression import PipelineConfig, apply
from magos.compression import registry as reg_mod


@dataclass
class _StubResult:
    messages: list[dict[str, Any]]
    tokens_before: int
    tokens_after: int
    transforms_applied: list[str] = field(default_factory=list)
    timing: dict[str, float] = field(default_factory=dict)
    waste_signals: Any | None = None


class _StubPipeline:
    def __init__(self, result: _StubResult) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def apply(self, **kwargs: Any) -> _StubResult:
        self.calls.append(kwargs)
        return self.result


def _patch_registry(monkeypatch: pytest.MonkeyPatch, pipeline: _StubPipeline) -> None:
    class _StubRegistry:
        def get_or_build(self, *_args: Any, **_kwargs: Any) -> _StubPipeline:
            return pipeline

        def pipelines(self) -> Any:
            return iter([pipeline])

    monkeypatch.setattr(reg_mod, "_REGISTRY", _StubRegistry())


def test_apply_returns_compressed_messages_when_savings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _StubPipeline(
        _StubResult(
            messages=[{"role": "user", "content": "short"}],
            tokens_before=100,
            tokens_after=60,
            transforms_applied=["ContentRouter"],
        )
    )
    _patch_registry(monkeypatch, pipeline)

    res = apply(
        messages=[{"role": "user", "content": "verbose original input"}],
        model="claude-sonnet-4-5",
        model_limit=200_000,
        config=PipelineConfig(),
        provider_name="anthropic",
    )

    assert res.messages == [{"role": "user", "content": "short"}]
    assert res.tokens_before == 100
    assert res.tokens_after == 60
    assert res.tokens_saved == 40
    assert res.transforms_applied == ["ContentRouter"]
    assert res.inflation_reverted is False


def test_apply_reverts_when_pipeline_inflates_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = [{"role": "user", "content": "input"}]
    pipeline = _StubPipeline(
        _StubResult(
            messages=[{"role": "user", "content": "MUCH LONGER OUTPUT"}],
            tokens_before=50,
            tokens_after=80,
            transforms_applied=["ContentRouter"],
        )
    )
    _patch_registry(monkeypatch, pipeline)

    res = apply(
        messages=original,
        model="claude-sonnet-4-5",
        model_limit=200_000,
        config=PipelineConfig(),
        provider_name="anthropic",
    )

    assert res.messages is original
    assert res.tokens_before == 50
    assert res.tokens_after == 50
    assert res.tokens_saved == 0
    assert res.inflation_reverted is True


def test_apply_passes_kwargs_through_to_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _StubPipeline(
        _StubResult(
            messages=[{"role": "user", "content": "x"}],
            tokens_before=10,
            tokens_after=8,
        )
    )
    _patch_registry(monkeypatch, pipeline)

    apply(
        messages=[{"role": "user", "content": "x"}],
        model="claude-sonnet-4-5",
        model_limit=128_000,
        config=PipelineConfig(),
        provider_name="anthropic",
        context="user query",
        biases={"foo": 1.0},
    )

    assert pipeline.calls == [
        {
            "messages": [{"role": "user", "content": "x"}],
            "model": "claude-sonnet-4-5",
            "model_limit": 128_000,
            "compress_user_messages": False,
            "compress_system_messages": True,
            "protect_recent": 4,
            "protect_analysis_context": True,
            "target_ratio": None,
            "min_tokens_to_compress": 250,
            "kompress_model": None,
            "frozen_message_count": 0,
            "context": "user query",
            "biases": {"foo": 1.0},
        }
    ]


def test_apply_forwards_compress_config_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _StubPipeline(
        _StubResult(
            messages=[{"role": "user", "content": "x"}],
            tokens_before=10,
            tokens_after=8,
        )
    )
    _patch_registry(monkeypatch, pipeline)

    apply(
        messages=[{"role": "user", "content": "x"}],
        model="claude-sonnet-4-5",
        model_limit=128_000,
        config=PipelineConfig(),
        provider_name="anthropic",
        compress_user_messages=True,
        protect_recent=0,
        target_ratio=0.5,
        min_tokens_to_compress=100,
        kompress_model="custom/model",
    )

    forwarded = pipeline.calls[0]
    assert forwarded["compress_user_messages"] is True
    assert forwarded["compress_system_messages"] is True  # default preserved
    assert forwarded["protect_recent"] == 0
    assert forwarded["target_ratio"] == 0.5
    assert forwarded["min_tokens_to_compress"] == 100
    assert forwarded["kompress_model"] == "custom/model"


def test_apply_forwards_frozen_message_count(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _StubPipeline(
        _StubResult(
            messages=[{"role": "user", "content": "x"}],
            tokens_before=10,
            tokens_after=8,
        )
    )
    _patch_registry(monkeypatch, pipeline)

    apply(
        messages=[{"role": "user", "content": "x"}],
        model="claude-sonnet-4-5",
        model_limit=128_000,
        config=PipelineConfig(),
        provider_name="anthropic",
        frozen_message_count=3,
    )

    assert pipeline.calls[0]["frozen_message_count"] == 3


def test_apply_omits_frozen_message_count_when_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default 0 is the no-freeze case; we still forward the value so
    transforms that read it always see the same key."""
    pipeline = _StubPipeline(
        _StubResult(
            messages=[{"role": "user", "content": "x"}],
            tokens_before=10,
            tokens_after=8,
        )
    )
    _patch_registry(monkeypatch, pipeline)

    apply(
        messages=[{"role": "user", "content": "x"}],
        model="claude-sonnet-4-5",
        model_limit=128_000,
        config=PipelineConfig(),
        provider_name="anthropic",
    )

    assert pipeline.calls[0]["frozen_message_count"] == 0
