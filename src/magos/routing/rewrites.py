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
import io
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from magos.registry.state import RegistryState
from magos.routing.jq_compat import evaluate_patch
from magos.routing.request import RoutedRequest
from magos.routing.schema import (
    AddHeader,
    Compress,
    CompressOptions,
    JqPatch,
    RemoveHeader,
    Rewrite,
    SetHeader,
    SetModel,
)
from magos.telemetry import get_logger

log = get_logger("magos.routing.rewrites")

# /v1/responses uses ``input`` (string or list of items), not ``messages``.
# Headroom's compress() expects ``messages``-shaped input; until an adapter
# exists we skip compression on the responses endpoint family.
_COMPRESS_SUPPORTED_ENDPOINTS: frozenset[str] = frozenset(
    {"/v1/messages", "/v1/messages/count_tokens", "/v1/chat/completions"}
)


# Headroom's hardcoded fallback when the caller doesn't supply model_limit
# (`compress.py:161`). Used as our last-resort default when LiteLLM doesn't
# recognise the dispatch model.
_DEFAULT_MODEL_LIMIT = 200_000

# Per-model context-window cache. Populated lazily on first compress call
# for each unique model id. Stores the resolved limit on success AND the
# fallback default on failure, so we don't re-trigger LiteLLM's noisy
# "model not mapped" stderr print on every request for an unknown model.
_MODEL_LIMIT_CACHE: dict[str, int] = {}


def _resolve_model_limit(
    dispatch_model: str,
    *,
    registry: RegistryState | None = None,
    default: int = _DEFAULT_MODEL_LIMIT,
) -> int:
    """Look up the max input-token window for ``dispatch_model``.

    Resolution order (highest first):

    1. ``registry`` if supplied AND the entry has ``context_size`` set —
       bypasses the LiteLLM call entirely. Lookup is by exact namespaced
       id, then by raw_id scan as a fallback for bare model strings.
    2. ``litellm.get_model_info`` — reads ``max_input_tokens`` (or
       ``max_tokens`` as a fallback). LiteLLM accepts both bare model
       names (``gpt-4o``) and provider-prefixed forms.
    3. ``default`` — Headroom's hardcoded 200_000 fallback.

    Result is cached per ``dispatch_model`` so subsequent requests for
    the same model are free and don't re-print LiteLLM's provider list.
    Registry hits skip the cache so context_size updates on refresh
    flow through immediately.
    """
    registry_limit = _registry_context_size(dispatch_model, registry)
    if registry_limit is not None:
        return registry_limit

    if dispatch_model in _MODEL_LIMIT_CACHE:
        return _MODEL_LIMIT_CACHE[dispatch_model]

    limit = default
    try:
        # LiteLLM prints a multi-line provider list to stderr on unknown
        # models. Suppress it; the routing log already captures the model.
        with contextlib.redirect_stderr(io.StringIO()):
            import litellm  # noqa: PLC0415

            info = litellm.get_model_info(dispatch_model)
    except Exception:
        info = None

    if isinstance(info, dict):
        for key in ("max_input_tokens", "max_tokens"):
            value = info.get(key)
            if isinstance(value, int) and value > 0:
                limit = value
                break

    _MODEL_LIMIT_CACHE[dispatch_model] = limit
    if limit == default:
        log.debug("compress.model_limit_default", dispatch_model=dispatch_model, limit=default)
    else:
        log.debug("compress.model_limit_resolved", dispatch_model=dispatch_model, limit=limit)
    return limit


def _registry_context_size(model: str, registry: RegistryState | None) -> int | None:
    """Return the registry's ``context_size`` for ``model`` if known.

    Tries an exact namespaced lookup first; if ``model`` matches a single
    raw_id across providers, that is also accepted (the routing layer
    typically normalizes to namespaced form before reaching here, but
    explicit-rule paths may pass bare ids).
    """
    if registry is None:
        return None
    direct = registry.get(model)
    if direct is not None and direct.context_size is not None:
        return direct.context_size
    matches = [e for e in registry.entries.values() if e.raw_id == model]
    if len(matches) == 1 and matches[0].context_size is not None:
        return matches[0].context_size
    return None


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


def apply_rewrites(
    req: RoutedRequest,
    rewrites: Sequence[Rewrite],
    *,
    registry: RegistryState | None = None,
) -> RoutedRequest:
    """Apply ``rewrites`` in list order; return a new RoutedRequest.

    Empty list returns ``req`` unchanged (same identity). Original headers
    and body are never mutated. ``registry`` is plumbed through to the
    compress rewrite so context_size resolution can prefer the registry.
    """
    if not rewrites:
        return req
    out = req
    for rw in rewrites:
        out = _apply_one(out, rw, registry=registry)
    return out


def _apply_one(  # noqa: PLR0911
    req: RoutedRequest, rw: Rewrite, *, registry: RegistryState | None = None
) -> RoutedRequest:
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
        return _apply_compress(req, rw.compress, registry=registry)
    raise TypeError(f"unhandled Rewrite variant: {type(rw).__name__}")


def _apply_compress(  # noqa: PLR0911
    req: RoutedRequest,
    opts: CompressOptions,
    *,
    registry: RegistryState | None = None,
) -> RoutedRequest:
    """Run Headroom compression against ``req.body``.

    Endpoint dispatch:

    - ``/v1/messages``, ``/v1/messages/count_tokens``, ``/v1/chat/completions``:
      operate on ``body['messages']`` (Anthropic / OpenAI Chat shape).
    - ``/v1/responses``: ``mode: cache`` only, operates on ``body['instructions']``.
      Token-mode compression of Responses ``input`` is not supported (different
      shape, no upstream Headroom path) and silently no-ops.
    - Other endpoints (``/v1/responses/{id}`` family, etc.): skipped.

    Failure mode: ``headroom.compress()`` already wraps its pipeline in
    try/except, returns the original messages on error, and emits an OTel
    failure metric. We do not double-wrap. We do, however, swallow import
    errors so a missing heavy extra (kompress weights, etc.) cannot take
    the proxy down — log + pass through.
    """
    if req.endpoint == "/v1/responses":
        return _apply_compress_responses(req, opts)

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

    # Per-rule override wins; otherwise auto-detect via LiteLLM's registry
    # so IntelligentContext fires at the right threshold and ContentRouter
    # scales pressure correctly for the actual destination model.
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


def _run_cache_aligner(messages: list[Any], model: str, *, endpoint: str) -> Any:
    """Shared CacheAligner runner used by chat (``messages``) and Responses
    (``instructions``) paths.

    Returns the ``TransformResult`` on success, or ``None`` if the deps
    couldn't load, the aligner declared no-op, or apply raised. Logs are
    attached to the ``endpoint`` for traceability.
    """
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

    # Headroom defaults ``CacheAlignerConfig.enabled=False`` (the transform is
    # opt-in); ``mode: cache`` is exactly that opt-in, so flip it on here.
    # ``use_dynamic_detector=True`` is Headroom's intended default — Tier 1
    # regex catches UUIDs, request IDs, sessions, ISO 8601 datetimes, and
    # high-entropy identifiers in addition to dates. The caller has already
    # invoked ``_preload_sentence_transformers`` to win the native-load race
    # on Windows.
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


def _apply_cache_aligner(req: RoutedRequest, messages: list[Any], model: str) -> RoutedRequest:
    """Run CacheAligner on chat-shape ``messages``."""
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
    """Cache-align the ``/v1/responses`` ``instructions`` field.

    Only ``mode: cache`` is supported on Responses, and only against the
    top-level ``instructions`` string (the OpenAI analogue of the chat
    ``system`` prompt). Token-mode compression of ``input`` is not
    supported: its shape (string-or-list-of-typed-items) doesn't round-
    trip cleanly through Headroom's ``messages``-shaped pipeline, and
    Headroom has no upstream Responses path of its own. Operators
    wanting that should compress the input before it reaches magos.
    """
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
    # Wrap the instructions string as a synthetic system message so the
    # CacheAligner's system-prompt branch fires. The aligner mutates the
    # message's ``content`` in place; we read it back and write it to the
    # ``instructions`` field. No new messages are introduced.
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


def _with_header(headers: Mapping[str, str], name: str, value: str) -> dict[str, str]:
    new_headers = dict(headers)
    new_headers[name.lower()] = value
    return new_headers
