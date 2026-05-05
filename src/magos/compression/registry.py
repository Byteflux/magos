"""Lazy per-(config, provider) registry of ``TransformPipeline`` instances."""

from __future__ import annotations

import threading
from collections.abc import Iterator

from headroom.transforms import TransformPipeline

from .build import ProviderName, build_pipeline
from .config import PipelineConfig


class PipelineRegistry:
    """Thread-safe lazy cache keyed by ``(fingerprint, provider_name)``."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], TransformPipeline] = {}
        self._lock = threading.Lock()

    def get_or_build(
        self, config: PipelineConfig, *, provider_name: ProviderName
    ) -> TransformPipeline:
        key = (config.fingerprint(), provider_name)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            pipeline = build_pipeline(config, provider_name=provider_name)
            self._cache[key] = pipeline
            return pipeline

    def pipelines(self) -> Iterator[TransformPipeline]:
        return iter(self._cache.values())


_REGISTRY = PipelineRegistry()


def get_registry() -> PipelineRegistry:
    return _REGISTRY
