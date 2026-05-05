"""Construct a ``TransformPipeline`` for a given ``PipelineConfig``.

CacheAligner is created **disabled**: prefix stability is the routing
layer's responsibility (see ``cache_mode.py`` for the standalone aligner
used by ``mode: cache``). The proxy follows the same convention; see
``docs/headroom/pipeline.md``.
"""

from __future__ import annotations

from typing import Literal

from headroom.config import (
    CacheAlignerConfig,
    IntelligentContextConfig,
    RollingWindowConfig,
)
from headroom.providers.anthropic import AnthropicProvider
from headroom.providers.base import Provider
from headroom.providers.openai import OpenAIProvider
from headroom.transforms import (
    CacheAligner,
    ContentRouter,
    ContentRouterConfig,
    IntelligentContextManager,
    RollingWindow,
    SmartCrusher,
    Transform,
    TransformPipeline,
)
from headroom.transforms.smart_crusher import SmartCrusherConfig

from .config import PipelineConfig

ProviderName = Literal["anthropic", "openai"]


def build_pipeline(config: PipelineConfig, *, provider_name: ProviderName) -> TransformPipeline:
    """Return a fresh ``TransformPipeline`` shaped by ``config``."""
    provider = _build_provider(provider_name)
    transforms: list[Transform] = [CacheAligner(CacheAlignerConfig(enabled=False))]

    if config.smart_routing:
        transforms.append(ContentRouter(ContentRouterConfig(enable_code_aware=config.code_aware)))
    else:
        transforms.append(SmartCrusher(SmartCrusherConfig(enabled=True)))

    if config.intelligent_context:
        transforms.append(
            IntelligentContextManager(
                IntelligentContextConfig(
                    enabled=True,
                    keep_system=True,
                    keep_last_turns=config.keep_last_turns,
                )
            )
        )
    else:
        transforms.append(
            RollingWindow(
                RollingWindowConfig(
                    enabled=True,
                    keep_system=True,
                    keep_last_turns=config.keep_last_turns,
                )
            )
        )

    return TransformPipeline(transforms=transforms, provider=provider)


def _build_provider(provider_name: str) -> Provider:
    if provider_name == "anthropic":
        return AnthropicProvider()
    if provider_name == "openai":
        return OpenAIProvider()
    raise ValueError(f"unknown provider_name: {provider_name!r}")
