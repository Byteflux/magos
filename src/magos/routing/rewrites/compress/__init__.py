"""Compress rewrite package.

`Compress.apply` (defined on the schema class in
`magos.routing.schema.rewrites`) dispatches to the appropriate
`magos.compression.engine` step. This package retains the
`model_limit` and `_preload` sub-modules which are imported
externally.
"""

from __future__ import annotations

from magos.routing.rewrites.compress import model_limit
from magos.routing.rewrites.compress._preload import _preload_sentence_transformers
from magos.routing.rewrites.compress.model_limit import (
    _DEFAULT_MODEL_LIMIT,
    _MODEL_LIMIT_CACHE,
    _resolve_model_limit,
)

__all__ = [
    "_DEFAULT_MODEL_LIMIT",
    "_MODEL_LIMIT_CACHE",
    "_preload_sentence_transformers",
    "_resolve_model_limit",
    "model_limit",
]
