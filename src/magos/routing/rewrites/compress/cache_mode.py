"""Cache-aligner mode for the compress rewrite.

Handles ``mode: cache`` for chat-shape ``messages`` endpoints and the
``/v1/responses`` ``instructions`` field via a synthetic-message wrapper.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any

from magos.routing.request import RoutedRequest
from magos.routing.schema import CompressOptions
from magos.telemetry import get_logger

from ._preload import _preload_sentence_transformers

log = get_logger("magos.routing.rewrites")


def _apply_cache_aligner(req: RoutedRequest, messages: list[Any], model: str) -> RoutedRequest:
    """CacheAligner on chat-shape ``messages``."""
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


def _apply_compress_responses(req: RoutedRequest, opts: CompressOptions) -> RoutedRequest:
    """Cache-align the ``/v1/responses`` ``instructions`` field; token mode unsupported."""
    if opts.mode != "cache":
        log.debug(
            "compress.responses_token_mode_unsupported",
            endpoint=req.endpoint,
            hint="use mode: cache to stabilise the instructions prefix",
        )
        return req

    instructions = req.body.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        return req

    model = str(req.body.get("model", "")) or "gpt-4o"
    # Wrap as a synthetic system message so the aligner's system-prompt
    # branch fires; we read the mutated content back into ``instructions``.
    synthetic = [{"role": "system", "content": instructions}]
    result = _run_cache_aligner(synthetic, model, endpoint=req.endpoint)
    if result is None:
        return req

    new_instructions = result.messages[0].get("content")
    if not isinstance(new_instructions, str) or new_instructions == instructions:
        return req

    log.info(
        "compress.applied",
        endpoint=req.endpoint,
        mode="cache",
        field="instructions",
        transforms=dict(Counter(result.transforms_applied)),
    )
    new_body = dict(req.body)
    new_body["instructions"] = new_instructions
    return replace(req, body=new_body, body_dirty=True)


def _run_cache_aligner(messages: list[Any], model: str, *, endpoint: str) -> Any:
    """Return the ``TransformResult``, or ``None`` on import / no-op / apply failure."""
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

    # Headroom defaults ``CacheAlignerConfig.enabled=False``; flip on for ``mode: cache``.
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
