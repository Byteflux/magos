"""Token-mode compression dispatch for the compress rewrite.

Calls ``magos.compression.apply``, which owns a process-wide registry of
``TransformPipeline`` instances bound per (config-fingerprint, provider).
``magos.compression`` enforces the inflation guard; this module's job is
to translate the routing-layer schema into a ``PipelineConfig`` and the
endpoint into a provider name.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any

from magos.compression import PipelineConfig, ProviderName, apply
from magos.registry.state import RegistryState
from magos.routing.request import RoutedRequest
from magos.routing.schema import CompressOptions
from magos.telemetry import get_logger

from ._preload import _preload_sentence_transformers
from .model_limit import _resolve_model_limit

log = get_logger("magos.routing.rewrites")

_OPENAI_ENDPOINTS: frozenset[str] = frozenset({"/v1/chat/completions"})


def _provider_for_endpoint(endpoint: str) -> ProviderName:
    return "openai" if endpoint in _OPENAI_ENDPOINTS else "anthropic"


def _apply_token_mode(
    req: RoutedRequest,
    messages: list[Any],
    opts: CompressOptions,
    *,
    registry: RegistryState | None = None,
) -> RoutedRequest:
    """Run the magos compression pipeline on ``messages``."""
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

    result = apply(
        messages=messages,
        model=model,
        model_limit=model_limit,
        config=config,
        provider_name=provider_name,
    )

    if result.inflation_reverted or result.tokens_saved <= 0:
        log.debug(
            "compress.no_savings",
            endpoint=req.endpoint,
            tokens_before=result.tokens_before,
            inflation_reverted=result.inflation_reverted,
        )
        return req

    log.info(
        "compress.applied",
        endpoint=req.endpoint,
        mode="token",
        provider=provider_name,
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
