"""Async pipeline: Anthropic Messages request -> OpenAI dispatch -> Anthropic response.

Pure function. The ``completion`` argument is the seam for tests and routing:
production wires ``litellm.acompletion``; tests inject a fake. Anything that
returns a dict-like or pydantic ``model_dump``-able response works.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol

import litellm

from magos.obs import get_logger, traced
from magos.tokens import count_locally, resolve_provider
from magos.translation import (
    AnthropicStreamTranslator,
    request_anthropic_to_openai,
    response_openai_to_anthropic,
)

log = get_logger("magos.proxy")


class _CompletionFn(Protocol):
    def __call__(self, **kwargs: Any) -> Awaitable[Any]: ...


def _coerce_to_dict(resp: Any) -> dict[str, Any]:
    if hasattr(resp, "model_dump"):
        dumped: dict[str, Any] = resp.model_dump()
        return dumped
    if isinstance(resp, dict):
        return dict(resp)
    raise TypeError(f"completion returned unsupported type: {type(resp).__name__}")


def _sse_event(data: str) -> bytes:
    return f"data: {data}\n\n".encode()


def _normalize_dispatch_payload(
    payload: dict[str, Any], forward_headers: dict[str, str] | None
) -> dict[str, Any]:
    """Add provider prefix to bare model names and merge forward_headers.

    LiteLLM rejects bare names like ``claude-3-5-sonnet-...`` without a
    ``<provider>/`` prefix; we infer the provider so the proxy works for
    clients that send the unprefixed names. ``forward_headers`` are merged
    into ``extra_headers`` so upstream sees client auth, version pins, and
    beta flags verbatim, preserving Anthropic's billing shape.
    """
    out = dict(payload)
    model = str(out.get("model", ""))
    if model and "/" not in model:
        provider = resolve_provider(model)
        if provider:
            out["model"] = f"{provider}/{model}"
    if forward_headers:
        existing = out.get("extra_headers") or {}
        out["extra_headers"] = {**existing, **forward_headers}
    return out


def _sse_named_event(event: dict[str, Any]) -> bytes:
    """Anthropic streaming uses ``event:`` + ``data:`` lines per chunk."""
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


@traced("proxy.anthropic_messages")
async def proxy_anthropic_messages(
    anthropic_request: dict[str, Any],
    *,
    completion: _CompletionFn | None = None,
    forward_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Round-trip an Anthropic Messages request through an OpenAI-shape upstream."""
    dispatch: Callable[..., Awaitable[Any]] = completion or litellm.acompletion
    openai_request = request_anthropic_to_openai(anthropic_request)
    payload = _normalize_dispatch_payload(openai_request, forward_headers)
    log.info("dispatch", shape="anthropic->openai", model=payload.get("model"))
    raw_response = await dispatch(**payload)
    openai_response = _coerce_to_dict(raw_response)
    return response_openai_to_anthropic(openai_response)


@traced("proxy.openai_chat_completions")
async def proxy_openai_chat_completions(
    openai_request: dict[str, Any],
    *,
    completion: _CompletionFn | None = None,
    forward_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Pass an OpenAI Chat Completions request through litellm without translation."""
    dispatch: Callable[..., Awaitable[Any]] = completion or litellm.acompletion
    payload = _normalize_dispatch_payload(openai_request, forward_headers)
    log.info("dispatch", shape="openai", model=payload.get("model"))
    raw_response = await dispatch(**payload)
    return _coerce_to_dict(raw_response)


def stream_anthropic_messages(
    anthropic_request: dict[str, Any],
    *,
    completion: _CompletionFn | None = None,
    forward_headers: dict[str, str] | None = None,
) -> AsyncIterator[bytes]:
    """Async iterator of Anthropic-shape SSE bytes for an Anthropic Messages request.

    Returned as a regular function (not an async generator) so request
    validation and the local token estimate run synchronously: a malformed
    request raises ``pydantic.ValidationError`` before any response bytes
    are emitted, and ``message_start.usage.input_tokens`` is seeded with a
    LiteLLM-based estimate so clients see a real value rather than ``0``.

    Streaming intentionally uses the local estimator only; provider passthrough
    would add network latency to time-to-first-byte. Anthropic clients refine
    usage from ``message_delta``, so an estimate at ``message_start`` is fine.
    """
    dispatch: Callable[..., Awaitable[Any]] = completion or litellm.acompletion
    openai_request = request_anthropic_to_openai(anthropic_request)
    try:
        input_tokens = count_locally(anthropic_request)
    except Exception as exc:
        log.warning("stream.token_count_failed", error=str(exc), error_type=type(exc).__name__)
        input_tokens = 0
    payload = _normalize_dispatch_payload({**openai_request, "stream": True}, forward_headers)
    log.info(
        "dispatch",
        shape="anthropic->openai",
        model=payload.get("model"),
        stream=True,
        input_tokens=input_tokens,
    )
    return _anthropic_stream_iter(payload, dispatch, input_tokens=input_tokens)


async def _anthropic_stream_iter(
    payload: dict[str, Any],
    dispatch: Callable[..., Awaitable[Any]],
    *,
    input_tokens: int = 0,
) -> AsyncIterator[bytes]:
    translator = AnthropicStreamTranslator(input_tokens=input_tokens)
    try:
        stream = await dispatch(**payload)
        async for chunk in stream:
            for event in translator.feed(_coerce_to_dict(chunk)):
                yield _sse_named_event(event)
    except Exception as exc:
        log.error(
            "stream.dispatch_failed",
            error=str(exc),
            error_type=type(exc).__name__,
            model=payload.get("model"),
        )
        # Emit an Anthropic-shape error event so the client can surface the
        # failure cleanly instead of seeing a truncated stream and retrying.
        yield _sse_named_event(
            {
                "type": "error",
                "error": {"type": "api_error", "message": f"{type(exc).__name__}: {exc}"},
            }
        )
        return
    for event in translator.finish():
        yield _sse_named_event(event)


async def stream_openai_chat_completions(
    openai_request: dict[str, Any],
    *,
    completion: _CompletionFn | None = None,
    forward_headers: dict[str, str] | None = None,
) -> AsyncIterator[bytes]:
    """Stream OpenAI Chat Completions chunks as SSE bytes.

    Forces ``stream=True`` on the upstream call. Each chunk is JSON-encoded into
    a ``data: ...`` SSE event; the stream terminates with ``data: [DONE]``,
    matching OpenAI's wire format so existing OpenAI clients work unchanged.
    """
    dispatch: Callable[..., Awaitable[Any]] = completion or litellm.acompletion
    request = _normalize_dispatch_payload({**openai_request, "stream": True}, forward_headers)
    log.info("dispatch", shape="openai", model=request.get("model"), stream=True)
    stream = await dispatch(**request)
    async for chunk in stream:
        yield _sse_event(json.dumps(_coerce_to_dict(chunk)))
    yield _sse_event("[DONE]")
