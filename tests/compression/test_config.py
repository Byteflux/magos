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
