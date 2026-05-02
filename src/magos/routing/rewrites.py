"""Pure mutators for the routing pipeline.

Each rewrite consumes a ``RoutedRequest`` and returns a new one. The frozen
dataclass forbids in-place mutation, so we copy ``headers`` and ``body``
defensively and use ``dataclasses.replace`` to produce successors. Body-
touching ops (``SetModel``, ``JqPatch``) flip ``body_dirty`` so the
dispatcher knows it must re-serialise instead of forwarding ``raw_body``
verbatim under passthrough.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from magos.obs import get_logger
from magos.routing.jq_compat import evaluate_patch
from magos.routing.models import (
    AddHeader,
    Compress,
    CompressOptions,
    JqPatch,
    RemoveHeader,
    Rewrite,
    SetHeader,
    SetModel,
)
from magos.routing.request import RoutedRequest

log = get_logger("magos.routing.rewrites")

# /v1/responses uses ``input`` (string or list of items), not ``messages``.
# Headroom's compress() expects ``messages``-shaped input; until an adapter
# exists we skip compression on the responses endpoint family.
_COMPRESS_SUPPORTED_ENDPOINTS: frozenset[str] = frozenset(
    {"/v1/messages", "/v1/messages/count_tokens", "/v1/chat/completions"}
)


def _preload_sentence_transformers() -> None:
    """Force-import ``sentence_transformers`` before any headroom import.

    Workaround for a Windows native-load order interaction: importing
    ``cryptography.hazmat.bindings._rust`` (transitively pulled by
    ``mitmproxy.http``) before ``sentence_transformers`` causes pyarrow's
    ``.pyd`` to segfault during ``create_module``. Loading
    sentence_transformers first lets the Arrow C++ runtime initialise
    before any PyO3 Rust runtime, which keeps Headroom's
    ``DynamicContentDetector`` import safe.

    Cost: ~6s on first call (one-shot, then cached in ``sys.modules``).
    Magos's main process does not transitively load cryptography at
    import time (verified for ``magos.server`` and ``litellm``), so as
    long as this fires before the first compress request, the order is
    safe. See ``docs/headroom.md`` for the full bisection.

    Silently no-ops if sentence_transformers isn't installed; the
    detector will then fail to initialise inside Headroom and fall back
    to the legacy regex path.
    """
    with contextlib.suppress(Exception):
        import sentence_transformers  # noqa: F401, PLC0415


class RewriteError(ValueError):
    """Raised when a rewrite cannot be applied (e.g., jq_patch shape error)."""


def apply_rewrites(req: RoutedRequest, rewrites: Sequence[Rewrite]) -> RoutedRequest:
    """Apply ``rewrites`` in list order; return a new RoutedRequest.

    Empty list returns ``req`` unchanged (same identity). Original headers
    and body are never mutated.
    """
    if not rewrites:
        return req
    out = req
    for rw in rewrites:
        out = _apply_one(out, rw)
    return out


def _apply_one(req: RoutedRequest, rw: Rewrite) -> RoutedRequest:  # noqa: PLR0911
    if isinstance(rw, SetModel):
        new_body = dict(req.body)
        new_body["model"] = rw.set_model
        return replace(req, body=new_body, body_dirty=True)
    if isinstance(rw, SetHeader):
        return replace(
            req, headers=_with_header(req.headers, rw.set_header.name, rw.set_header.value)
        )
    if isinstance(rw, AddHeader):
        key = rw.add_header.name.lower()
        if key in req.headers:
            return req
        return replace(
            req, headers=_with_header(req.headers, rw.add_header.name, rw.add_header.value)
        )
    if isinstance(rw, RemoveHeader):
        key = rw.remove_header.lower()
        if key not in req.headers:
            return req
        new_headers = dict(req.headers)
        del new_headers[key]
        return replace(req, headers=new_headers)
    if isinstance(rw, JqPatch):
        result: Any = evaluate_patch(rw.jq_patch, dict(req.body))
        if not isinstance(result, Mapping):
            raise RewriteError(
                f"jq_patch result must be a JSON object, got "
                f"{type(result).__name__}: {rw.jq_patch!r}"
            )
        return replace(req, body=dict(result), body_dirty=True)
    if isinstance(rw, Compress):
        return _apply_compress(req, rw.compress)
    raise TypeError(f"unhandled Rewrite variant: {type(rw).__name__}")


def _apply_compress(req: RoutedRequest, opts: CompressOptions) -> RoutedRequest:
    """Run Headroom compression against ``req.body['messages']``.

    Endpoint scope: only Anthropic Messages and OpenAI Chat Completions
    have a top-level ``messages`` array. The Responses family uses
    ``input`` and is skipped here (returns ``req`` unchanged).

    Failure mode: ``headroom.compress()`` already wraps its pipeline in
    try/except, returns the original messages on error, and emits an OTel
    failure metric. We do not double-wrap. We do, however, swallow import
    errors so a missing heavy extra (kompress weights, etc.) cannot take
    the proxy down — log + pass through.
    """
    if req.endpoint not in _COMPRESS_SUPPORTED_ENDPOINTS:
        log.debug("compress.skipped_endpoint", endpoint=req.endpoint)
        return req

    messages = req.body.get("messages")
    if not isinstance(messages, list) or not messages:
        return req

    # Lazy import: headroom transitively pulls heavy deps (tokenizers,
    # optional sklearn/sentence-transformers); only pay that cost on rules
    # that actually use compress. Preload sentence_transformers first to
    # win the native-load order race on Windows.
    _preload_sentence_transformers()
    try:
        from headroom import compress as _hr_compress  # noqa: PLC0415
        from headroom.compress import CompressConfig  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("compress.import_failed", error=str(exc), error_type=type(exc).__name__)
        return req

    model = str(req.body.get("model", "")) or "claude-sonnet-4-5-20250929"

    if opts.mode == "cache":
        return _apply_cache_aligner(req, messages, model)

    cfg = CompressConfig(
        compress_user_messages=opts.compress_user_messages,
        compress_system_messages=opts.compress_system_messages,
        protect_recent=opts.protect_recent,
        protect_analysis_context=opts.protect_analysis_context,
        target_ratio=opts.target_ratio,
        min_tokens_to_compress=opts.min_tokens_to_compress,
        kompress_model=opts.kompress_model,
    )
    result = _hr_compress(messages, model=model, config=cfg)

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
        transforms=result.transforms_applied,
    )
    new_body = dict(req.body)
    new_body["messages"] = result.messages
    return replace(req, body=new_body, body_dirty=True)


def _apply_cache_aligner(req: RoutedRequest, messages: list[Any], model: str) -> RoutedRequest:
    """Run only CacheAligner — stabilise the prefix, do not compress."""
    _preload_sentence_transformers()
    try:
        from headroom.config import CacheAlignerConfig  # noqa: PLC0415
        from headroom.tokenizer import Tokenizer  # noqa: PLC0415
        from headroom.tokenizers import EstimatingTokenCounter  # noqa: PLC0415
        from headroom.transforms.cache_aligner import CacheAligner  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover
        log.warning(
            "compress.cache_align_import_failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return req

    # Headroom defaults ``CacheAlignerConfig.enabled=False`` (the transform is
    # opt-in); ``mode: cache`` is exactly that opt-in, so flip it on here.
    # ``use_dynamic_detector=True`` is Headroom's intended default — Tier 1
    # regex catches UUIDs, request IDs, sessions, ISO 8601 datetimes, and
    # high-entropy identifiers in addition to dates. ``_preload_sentence_transformers``
    # already ran above to neutralise the Windows native-load order bug.
    aligner = CacheAligner(CacheAlignerConfig(enabled=True))
    tokenizer = Tokenizer(EstimatingTokenCounter(), model=model)
    if not aligner.should_apply(messages, tokenizer, model=model):
        log.debug("compress.cache_align_noop", endpoint=req.endpoint)
        return req

    try:
        result = aligner.apply(messages, tokenizer, model=model)
    except Exception as exc:
        log.warning(
            "compress.cache_align_failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return req

    log.info(
        "compress.applied",
        endpoint=req.endpoint,
        mode="cache",
        transforms=list(result.transforms_applied),
    )
    new_body = dict(req.body)
    new_body["messages"] = result.messages
    return replace(req, body=new_body, body_dirty=True)


def _with_header(headers: Mapping[str, str], name: str, value: str) -> dict[str, str]:
    new_headers = dict(headers)
    new_headers[name.lower()] = value
    return new_headers
