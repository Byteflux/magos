"""Headroom-driven context compression as a routing rewrite.

Two modes (``token`` / ``cache``) and per-endpoint scoping; see
``docs/headroom/pipeline.md`` for the dispatch matrix and failure modes.
Heavy optional deps (kompress, sentence_transformers) are import-guarded.
"""

from __future__ import annotations

from magos.registry.state import RegistryState
from magos.routing.request import RoutedRequest
from magos.routing.schema import Compress, CompressOptions
from magos.telemetry import get_logger

from ._preload import _preload_sentence_transformers
from .cache_mode import _apply_cache_aligner, _apply_compress_responses
from .model_limit import _DEFAULT_MODEL_LIMIT, _MODEL_LIMIT_CACHE, _resolve_model_limit
from .token_mode import _apply_token_mode

# Public surface consumed by callers:
#   apply_compress        - ingress/http/run.py, routing/rewrites/__init__.py
#   _preload_sentence_transformers - cli/serve.py, ingress/http/lifespan.py
#   _MODEL_LIMIT_CACHE    - tests (monkeypatched)
#   _resolve_model_limit  - tests, test_compress_registry.py
#   _DEFAULT_MODEL_LIMIT  - tests

__all__ = [
    "_DEFAULT_MODEL_LIMIT",
    "_MODEL_LIMIT_CACHE",
    "_preload_sentence_transformers",
    "_resolve_model_limit",
    "apply_compress",
]

log = get_logger("magos.routing.rewrites")

# Endpoints whose body has a ``messages`` array compatible with Headroom's
# pipeline. /v1/responses uses ``input`` instead and is handled separately.
_COMPRESS_SUPPORTED_ENDPOINTS: frozenset[str] = frozenset(
    {"/v1/messages", "/v1/messages/count_tokens", "/v1/chat/completions"}
)


def apply_compress(
    req: RoutedRequest, rw: Compress, *, registry: RegistryState | None = None
) -> RoutedRequest:
    """Dispatch by endpoint and mode."""
    return _apply_compress(req, rw.compress, registry=registry)


def _apply_compress(
    req: RoutedRequest,
    opts: CompressOptions,
    *,
    registry: RegistryState | None = None,
) -> RoutedRequest:
    if req.endpoint == "/v1/responses":
        return _apply_compress_responses(req, opts)

    if req.endpoint not in _COMPRESS_SUPPORTED_ENDPOINTS:
        log.debug("compress.skipped_endpoint", endpoint=req.endpoint)
        return req

    messages = req.body.get("messages")
    if not isinstance(messages, list) or not messages:
        return req

    if opts.mode == "cache":
        model = str(req.body.get("model", "")) or "claude-sonnet-4-5-20250929"
        return _apply_cache_aligner(req, messages, model)

    return _apply_token_mode(req, messages, opts, registry=registry)
