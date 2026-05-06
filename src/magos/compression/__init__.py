"""Magos-owned compression pipeline layer.

Wraps ``headroom.transforms.TransformPipeline`` with a per-(config, provider)
registry, eager warmup, and an inflation guard. The ``compress`` routing
rewrite calls ``apply`` from this module instead of ``headroom.compress``.
"""

from __future__ import annotations

from .build import ProviderName, build_pipeline
from .config import PipelineConfig, pipeline_config_from_compress_options
from .pipeline import ApplyResult, apply
from .registry import PipelineRegistry, get_registry
from .warmup import eager_warmup

__all__ = [
    "ApplyResult",
    "PipelineConfig",
    "PipelineRegistry",
    "ProviderName",
    "apply",
    "build_pipeline",
    "eager_warmup",
    "get_registry",
    "pipeline_config_from_compress_options",
]
