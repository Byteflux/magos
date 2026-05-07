"""Token-mode compression engine step.

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

from headroom.ccr import CCRToolInjector

from magos.cache import PrefixCacheTracker, derive_session_id, get_store
from magos.compression import ProviderName, apply, pipeline_config_from_compress_options
from magos.compression.engine.base import Compressor
from magos.registry.state import RegistryState
from magos.routing.request import PostResponseHook, RoutedRequest
from magos.routing.rewrites.compress.model_limit import _resolve_model_limit
from magos.routing.schema.rewrites import CompressOptions
from magos.shapes import Usage
from magos.telemetry import get_logger

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


class TokenCompressor(Compressor):
    """Run the magos compression pipeline on ``messages`` with prefix-cache awareness."""

    def __init__(self, opts: CompressOptions) -> None:
        self._opts = opts

    def apply(
        self,
        req: RoutedRequest,
        *,
        registry: RegistryState | None = None,
    ) -> RoutedRequest:
        from magos.routing.rewrites.compress._preload import (  # noqa: PLC0415
            _preload_sentence_transformers,
        )

        _preload_sentence_transformers()

        opts = self._opts
        messages: list[Any] = list(req.body.get("messages", []))

        model = str(req.body.get("model", "")) or "claude-sonnet-4-5-20250929"
        model_limit = (
            opts.model_limit
            if opts.model_limit is not None
            else _resolve_model_limit(model, registry=registry)
        )

        config = pipeline_config_from_compress_options(opts)
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

        # CCR tool injection: when post-compression messages carry compression
        # markers, inject ``headroom_retrieve`` so the model can retrieve the
        # original content. Frozen prefix > 0 disables instruction injection
        # to preserve prefix cache; tool injection still runs.
        new_tools: list[dict[str, Any]] | None = None
        new_messages_for_ccr: list[dict[str, Any]] | None = None
        if opts.ccr_enabled:
            instructions_enabled = opts.ccr_inject_instructions and frozen_count == 0
            injector = CCRToolInjector(
                provider=provider_name,
                inject_tool=opts.ccr_inject_tool,
                inject_system_instructions=instructions_enabled,
            )
            existing_tools_raw = req.body.get("tools")
            existing_tools = (
                list(existing_tools_raw) if isinstance(existing_tools_raw, list) else None
            )
            injected_messages, injected_tools, was_injected = injector.process_request(
                list(result.messages), existing_tools
            )
            if injector.has_compressed_content:
                new_tools = injected_tools
                new_messages_for_ccr = injected_messages
                log.info(
                    "ccr.injected",
                    endpoint=req.endpoint,
                    provider=provider_name,
                    hashes=injector.detected_hashes,
                    tool_was_injected=was_injected,
                    instructions_skipped_for_frozen_prefix=(
                        bool(opts.ccr_inject_instructions and frozen_count > 0)
                    ),
                )

        if result.inflation_reverted or result.tokens_saved <= 0:
            log.debug(
                "compress.no_savings",
                endpoint=req.endpoint,
                tokens_before=result.tokens_before,
                inflation_reverted=result.inflation_reverted,
                session_id=session_id,
                frozen_count=frozen_count,
            )
            if new_messages_for_ccr is not None or new_tools is not None:
                new_body = dict(req.body)
                if new_messages_for_ccr is not None:
                    new_body["messages"] = new_messages_for_ccr
                if new_tools is not None:
                    new_body["tools"] = new_tools
                return replace(req, body=new_body, body_dirty=True)
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
        new_body["messages"] = (
            new_messages_for_ccr if new_messages_for_ccr is not None else result.messages
        )
        if new_tools is not None:
            new_body["tools"] = new_tools
        return replace(req, body=new_body, body_dirty=True)
