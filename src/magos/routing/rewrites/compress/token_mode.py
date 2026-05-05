"""Token-mode compression dispatch for the compress rewrite."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any

from magos.registry.state import RegistryState
from magos.routing.request import RoutedRequest
from magos.routing.schema import CompressOptions
from magos.telemetry import get_logger

from ._preload import _preload_sentence_transformers
from .model_limit import _resolve_model_limit

log = get_logger("magos.routing.rewrites")


def _apply_token_mode(
    req: RoutedRequest,
    messages: list[Any],
    opts: CompressOptions,
    *,
    registry: RegistryState | None = None,
) -> RoutedRequest:
    """Run the full Headroom compress pipeline on ``messages``."""
    # Lazy import: headroom pulls heavy deps; only pay the cost when used.
    # Preload sentence_transformers first to win the Windows native-load race
    # (see ``docs/headroom/pipeline.md``).
    _preload_sentence_transformers()
    try:
        from headroom import compress as _hr_compress  # noqa: PLC0415
        from headroom.compress import CompressConfig  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("compress.import_failed", error=str(exc), error_type=type(exc).__name__)
        return req

    model = str(req.body.get("model", "")) or "claude-sonnet-4-5-20250929"

    # Per-rule override wins; else auto-detect so transforms fire at the
    # right threshold for the destination model.
    model_limit = (
        opts.model_limit
        if opts.model_limit is not None
        else _resolve_model_limit(model, registry=registry)
    )

    cfg = CompressConfig(
        compress_user_messages=opts.compress_user_messages,
        compress_system_messages=opts.compress_system_messages,
        protect_recent=opts.protect_recent,
        protect_analysis_context=opts.protect_analysis_context,
        target_ratio=opts.target_ratio,
        min_tokens_to_compress=opts.min_tokens_to_compress,
        kompress_model=opts.kompress_model,
    )
    result = _hr_compress(messages, model=model, model_limit=model_limit, config=cfg)

    if result.tokens_saved <= 0:
        log.debug(
            "compress.no_savings",
            endpoint=req.endpoint,
            tokens_before=result.tokens_before,
        )
        return req

    log.info(
        "compress.applied",
        endpoint=req.endpoint,
        mode="token",
        tokens_before=result.tokens_before,
        tokens_after=result.tokens_after,
        tokens_saved=result.tokens_saved,
        ratio=round(result.compression_ratio, 4),
        transforms=dict(Counter(result.transforms_applied)),
    )
    new_body = dict(req.body)
    new_body["messages"] = result.messages
    return replace(req, body=new_body, body_dirty=True)
