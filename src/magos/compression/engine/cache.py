"""Cache-aligner mode compression engine step.

Handles `mode: cache` for chat-shape `messages` endpoints.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any

from magos.compression.engine.base import Compressor
from magos.registry.state import RegistryState
from magos.routing.request import RoutedRequest
from magos.routing.schema.rewrites import CompressOptions
from magos.telemetry import get_logger

log = get_logger("magos.routing.rewrites")


def _run_cache_aligner(messages: list[Any], model: str, *, endpoint: str) -> Any:
    """Return the `TransformResult`, or `None` on import / no-op / apply failure."""
    from magos.routing.rewrites.compress._preload import (  # noqa: PLC0415
        _preload_sentence_transformers,
    )

    _preload_sentence_transformers()
    try:
        from headroom.config import CacheAlignerConfig  # noqa: PLC0415
        from headroom.tokenizer import Tokenizer  # noqa: PLC0415
        from headroom.tokenizers import EstimatingTokenCounter  # noqa: PLC0415
        from headroom.transforms.cache_aligner import CacheAligner  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover
        log.warning(
            "compress.cache_align_import_failed",
            endpoint=endpoint,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return None

    # Headroom defaults `CacheAlignerConfig.enabled=False`; flip on for `mode: cache`.
    aligner = CacheAligner(CacheAlignerConfig(enabled=True))
    tokenizer = Tokenizer(EstimatingTokenCounter(), model=model)
    if not aligner.should_apply(messages, tokenizer, model=model):
        log.debug("compress.cache_align_noop", endpoint=endpoint)
        return None

    try:
        return aligner.apply(messages, tokenizer, model=model)
    except Exception as exc:
        log.warning(
            "compress.cache_align_failed",
            endpoint=endpoint,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return None


class CacheCompressor(Compressor):
    """CacheAligner on chat-shape `messages`."""

    def __init__(self, opts: CompressOptions) -> None:
        self._opts = opts

    def apply(
        self,
        req: RoutedRequest,
        *,
        registry: RegistryState | None = None,
    ) -> RoutedRequest:
        messages: list[Any] = list(req.body.get("messages", []))
        model = str(req.body.get("model", "")) or "claude-sonnet-4-5-20250929"

        result = _run_cache_aligner(messages, model, endpoint=req.endpoint)
        if result is None:
            return req

        log.info(
            "compress.applied",
            endpoint=req.endpoint,
            mode="cache",
            transforms=dict(Counter(result.transforms_applied)),
        )
        new_body = dict(req.body)
        new_body["messages"] = result.messages
        return replace(req, body=new_body, body_dirty=True)
