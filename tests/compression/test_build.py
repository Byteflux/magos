"""`build_pipeline` produces the right transform shape for each config."""

from __future__ import annotations

import pytest
from headroom.transforms import (
    CacheAligner,
    ContentRouter,
    IntelligentContextManager,
    RollingWindow,
    SmartCrusher,
    TransformPipeline,
)

from magos.compression import PipelineConfig
from magos.compression.build import build_pipeline


def test_default_config_smart_routing_intelligent_context() -> None:
    pipeline = build_pipeline(PipelineConfig(), provider_name="anthropic")
    assert isinstance(pipeline, TransformPipeline)
    types = [type(t) for t in pipeline.transforms]
    assert types == [CacheAligner, ContentRouter, IntelligentContextManager]


def test_legacy_routing_uses_smart_crusher() -> None:
    cfg = PipelineConfig(smart_routing=False)
    pipeline = build_pipeline(cfg, provider_name="anthropic")
    types = [type(t) for t in pipeline.transforms]
    assert types == [CacheAligner, SmartCrusher, IntelligentContextManager]


def test_rolling_window_replaces_intelligent_context() -> None:
    cfg = PipelineConfig(intelligent_context=False)
    pipeline = build_pipeline(cfg, provider_name="openai")
    types = [type(t) for t in pipeline.transforms]
    assert types == [CacheAligner, ContentRouter, RollingWindow]


def test_cache_aligner_is_disabled_in_pipeline() -> None:
    pipeline = build_pipeline(PipelineConfig(), provider_name="anthropic")
    aligner = pipeline.transforms[0]
    assert isinstance(aligner, CacheAligner)
    assert aligner.config.enabled is False


def test_keep_last_turns_threads_into_context_manager() -> None:
    cfg = PipelineConfig(intelligent_context=False, keep_last_turns=7)
    pipeline = build_pipeline(cfg, provider_name="anthropic")
    rw = pipeline.transforms[-1]
    assert isinstance(rw, RollingWindow)
    assert rw.config.keep_last_turns == 7


def test_anthropic_provider_is_bound() -> None:
    pipeline = build_pipeline(PipelineConfig(), provider_name="anthropic")
    assert pipeline._provider.name == "anthropic"  # type: ignore[union-attr]


def test_openai_provider_is_bound() -> None:
    pipeline = build_pipeline(PipelineConfig(), provider_name="openai")
    assert pipeline._provider.name == "openai"  # type: ignore[union-attr]


def test_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="provider_name"):
        build_pipeline(PipelineConfig(), provider_name="azure")  # type: ignore[arg-type]
