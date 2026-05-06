"""``PipelineConfig`` value semantics + fingerprinting."""

from __future__ import annotations

import dataclasses

import pytest

from magos.compression import PipelineConfig


def test_pipeline_config_defaults_match_proxy_modern_shape() -> None:
    cfg = PipelineConfig()
    assert cfg.smart_routing is True
    assert cfg.code_aware is False
    assert cfg.intelligent_context is True
    assert cfg.keep_last_turns == 4


def test_pipeline_config_is_frozen() -> None:
    cfg = PipelineConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.smart_routing = False  # type: ignore[misc]


def test_fingerprint_is_stable_for_equal_configs() -> None:
    a = PipelineConfig(smart_routing=True, code_aware=False)
    b = PipelineConfig(smart_routing=True, code_aware=False)
    assert a.fingerprint() == b.fingerprint()


def test_fingerprint_differs_for_distinct_configs() -> None:
    a = PipelineConfig(smart_routing=True)
    b = PipelineConfig(smart_routing=False)
    assert a.fingerprint() != b.fingerprint()


def test_pipeline_config_from_compress_options_extracts_runtime_knobs() -> None:
    from magos.compression import pipeline_config_from_compress_options  # noqa: PLC0415
    from magos.routing.schema import CompressOptions  # noqa: PLC0415

    opts = CompressOptions(
        smart_routing=False,
        code_aware=True,
        intelligent_context=False,
        keep_last_turns=8,
    )
    pc = pipeline_config_from_compress_options(opts)
    assert pc.smart_routing is False
    assert pc.code_aware is True
    assert pc.intelligent_context is False
    assert pc.keep_last_turns == 8


def test_pipeline_config_from_compress_options_ignores_non_pipeline_knobs() -> None:
    """target_ratio, kompress_model, etc. are pipeline.apply kwargs, not
    PipelineConfig fields. The transcoder must not look at them."""
    from magos.compression import pipeline_config_from_compress_options  # noqa: PLC0415
    from magos.routing.schema import CompressOptions  # noqa: PLC0415

    opts_a = CompressOptions(target_ratio=0.5, kompress_model="custom/m")
    opts_b = CompressOptions(target_ratio=0.9, kompress_model=None)
    pc_a = pipeline_config_from_compress_options(opts_a)
    pc_b = pipeline_config_from_compress_options(opts_b)
    assert pc_a.fingerprint() == pc_b.fingerprint()


def test_pipeline_config_from_compress_options_default_matches_pipelineconfig_default() -> None:
    """A default CompressOptions must transcode to the default PipelineConfig
    so that the existing default-config warmup path is unchanged."""
    from magos.compression import (  # noqa: PLC0415
        PipelineConfig,
        pipeline_config_from_compress_options,
    )
    from magos.routing.schema import CompressOptions  # noqa: PLC0415

    pc = pipeline_config_from_compress_options(CompressOptions())
    assert pc.fingerprint() == PipelineConfig().fingerprint()
