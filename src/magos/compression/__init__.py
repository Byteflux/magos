"""Magos-owned compression pipeline layer.

Wraps ``headroom.transforms.TransformPipeline`` with a per-(config, provider)
registry, eager warmup, and an inflation guard. The ``compress`` routing
rewrite calls ``apply`` from this module instead of ``headroom.compress``.

See ``docs/superpowers/plans/2026-05-05-compression-pipeline-ownership.md``.
"""

from __future__ import annotations

from .build import ProviderName, build_pipeline
from .config import PipelineConfig

__all__ = ["PipelineConfig", "ProviderName", "build_pipeline"]
