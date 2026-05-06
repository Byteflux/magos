"""Frozen configuration for a ``TransformPipeline`` instance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from magos.routing.schema import CompressOptions


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Inputs to ``build_pipeline`` that determine the transform shape.

    Defaults match Headroom's modern proxy default shape:
    ``CacheAligner(disabled) -> ContentRouter -> IntelligentContextManager``.
    """

    smart_routing: bool = True
    code_aware: bool = False
    intelligent_context: bool = True
    keep_last_turns: int = 4

    def fingerprint(self) -> str:
        """Stable hex digest used as a registry key."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()


def pipeline_config_from_compress_options(opts: CompressOptions) -> PipelineConfig:
    """Map ``CompressOptions`` runtime knobs to a ``PipelineConfig``.

    Single source of truth for the transcoding so warmup and runtime
    build identical fingerprints. ``CompressOptions`` carries additional
    knobs (``target_ratio``, ``kompress_model``, etc.) that flow into
    ``pipeline.apply`` kwargs rather than the pipeline's transform shape;
    those are intentionally ignored here.
    """
    return PipelineConfig(
        smart_routing=opts.smart_routing,
        code_aware=opts.code_aware,
        intelligent_context=opts.intelligent_context,
        keep_last_turns=opts.keep_last_turns,
    )
