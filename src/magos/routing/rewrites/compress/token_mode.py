"""Token-mode compression dispatch for the compress rewrite.

Calls ``magos.compression.apply`` after fetching a per-session
``PrefixCacheTracker`` from the ``magos.cache`` store, so the pipeline
knows how many leading messages are already cached upstream and must
not be modified. Also appends a ``post_response_hook`` that feeds the
upstream's reported cache_read / cache_write tokens back into the
tracker on the way out.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any

from magos.cache import derive_session_id, get_store
from magos.cache.tracker import PrefixCacheTracker
from magos.compression import PipelineConfig, ProviderName, apply
from magos.egress.usage import Usage
from magos.registry.state import RegistryState
from magos.routing.request import PostResponseHook, RoutedRequest
from magos.routing.schema import CompressOptions
from magos.telemetry import get_logger

from ._preload import _preload_sentence_transformers
from .model_limit import _resolve_model_limit

log = get_logger("magos.routing.rewrites")

_OPENAI_ENDPOINTS: frozenset[str] = frozenset({"/v1/chat/completions"})


def _provider_for_endpoint(endpoint: str) -> ProviderName:
    return "openai" if endpoint in _OPENAI_ENDPOINTS else "anthropic"


def _make_post_response_hook(
    tracker: PrefixCacheTracker,
    sent_messages: list[dict[str, Any]],
) -> PostResponseHook:
    """Closure that updates the tracker with cache_read/write from Usage."""

    def hook(usage: Usage) -> None:
        tracker.update_from_response(
            cache_read_tokens=usage.cache_read,
            cache_write_tokens=usage.cache_write,
            messages=sent_messages,
        )

    return hook


def _apply_token_mode(
    req: RoutedRequest,
    messages: list[Any],
    opts: CompressOptions,
    *,
    registry: RegistryState | None = None,
) -> RoutedRequest:
    """Run the magos compression pipeline on ``messages`` with prefix-cache awareness."""
    _preload_sentence_transformers()

    model = str(req.body.get("model", "")) or "claude-sonnet-4-5-20250929"
    model_limit = (
        opts.model_limit
        if opts.model_limit is not None
        else _resolve_model_limit(model, registry=registry)
    )

    config = PipelineConfig(
        smart_routing=opts.smart_routing,
        code_aware=opts.code_aware,
        intelligent_context=opts.intelligent_context,
        keep_last_turns=opts.keep_last_turns,
    )
    provider_name = _provider_for_endpoint(req.endpoint)

    session_id = derive_session_id(req.headers, req.body, provider_name)
    tracker = get_store().get_or_create(session_id, provider_name)
    frozen_count = tracker.get_frozen_message_count()

    result = apply(
        messages=messages,
        model=model,
        model_limit=model_limit,
        config=config,
        provider_name=provider_name,
        compress_user_messages=opts.compress_user_messages,
        compress_system_messages=opts.compress_system_messages,
        protect_recent=opts.protect_recent,
        protect_analysis_context=opts.protect_analysis_context,
        target_ratio=opts.target_ratio,
        min_tokens_to_compress=opts.min_tokens_to_compress,
        kompress_model=opts.kompress_model,
        frozen_message_count=frozen_count,
    )

    # Always register the hook, even on no-savings / inflation revert,
    # so the tracker observes upstream cache state for the next turn.
    sent_messages = list(result.messages)
    req.post_response_hooks.append(_make_post_response_hook(tracker, sent_messages))

    if result.inflation_reverted or result.tokens_saved <= 0:
        log.debug(
            "compress.no_savings",
            endpoint=req.endpoint,
            tokens_before=result.tokens_before,
            inflation_reverted=result.inflation_reverted,
            session_id=session_id,
            frozen_count=frozen_count,
        )
        return req

    log.info(
        "compress.applied",
        endpoint=req.endpoint,
        mode="token",
        provider=provider_name,
        session_id=session_id,
        frozen_count=frozen_count,
        tokens_before=result.tokens_before,
        tokens_after=result.tokens_after,
        tokens_saved=result.tokens_saved,
        ratio=round(
            result.tokens_saved / result.tokens_before if result.tokens_before > 0 else 0.0,
            4,
        ),
        transforms=dict(Counter(result.transforms_applied)),
    )
    new_body = dict(req.body)
    new_body["messages"] = result.messages
    return replace(req, body=new_body, body_dirty=True)
