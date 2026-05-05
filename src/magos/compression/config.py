"""Frozen configuration for a ``TransformPipeline`` instance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


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
