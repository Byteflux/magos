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


def _sse_named_event(event: dict[str, Any]) -> bytes:
    """Anthropic streaming uses ``event:`` + ``data:`` lines per chunk."""
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


@traced("proxy.anthropic_messages")
async def proxy_anthropic_messages(
    anthropic_request: dict[str, Any],
    *,
    completion: _CompletionFn | None = None,
) -> dict[str, Any]:
    """Round-trip an Anthropic Messages request through an OpenAI-shape upstream."""
    dispatch: Callable[..., Awaitable[Any]] = completion or litellm.acompletion
    openai_request = request_anthropic_to_openai(anthropic_request)
    log.info("dispatch", shape="anthropic->openai", model=openai_request.get("model"))
    raw_response = await dispatch(**openai_request)
    openai_response = _coerce_to_dict(raw_response)
    return response_openai_to_anthropic(openai_response)


@traced("proxy.openai_chat_completions")
async def proxy_openai_chat_completions(
    openai_request: dict[str, Any],
    *,
    completion: _CompletionFn | None = None,
) -> dict[str, Any]:
    """Pass an OpenAI Chat Completions request through litellm without translation."""
    dispatch: Callable[..., Awaitable[Any]] = completion or litellm.acompletion
    log.info("dispatch", shape="openai", model=openai_request.get("model"))
    raw_response = await dispatch(**openai_request)
    return _coerce_to_dict(raw_response)


def stream_anthropic_messages(
    anthropic_request: dict[str, Any],
    *,
    completion: _CompletionFn | None = None,
) -> AsyncIterator[bytes]:
    """Async iterator of Anthropic-shape SSE bytes for an Anthropic Messages request.

    Returned as a regular function (not an async generator) so request
    validation runs synchronously: a malformed request raises
    ``pydantic.ValidationError`` before any response bytes are emitted, which
    lets the endpoint return a clean 400 instead of a half-streamed reply.
    """
    dispatch: Callable[..., Awaitable[Any]] = completion or litellm.acompletion
    openai_request = request_anthropic_to_openai(anthropic_request)
    payload = {**openai_request, "stream": True}
    log.info("dispatch", shape="anthropic->openai", model=payload.get("model"), stream=True)
    return _anthropic_stream_iter(payload, dispatch)


async def _anthropic_stream_iter(
    payload: dict[str, Any],
    dispatch: Callable[..., Awaitable[Any]],
) -> AsyncIterator[bytes]:
    stream = await dispatch(**payload)
    translator = AnthropicStreamTranslator()
    async for chunk in stream:
        for event in translator.feed(_coerce_to_dict(chunk)):
            yield _sse_named_event(event)
    for event in translator.finish():
        yield _sse_named_event(event)


async def stream_openai_chat_completions(
    openai_request: dict[str, Any],
    *,
    completion: _CompletionFn | None = None,
) -> AsyncIterator[bytes]:
    """Stream OpenAI Chat Completions chunks as SSE bytes.

    Forces ``stream=True`` on the upstream call. Each chunk is JSON-encoded into
    a ``data: ...`` SSE event; the stream terminates with ``data: [DONE]``,
    matching OpenAI's wire format so existing OpenAI clients work unchanged.
    """
    dispatch: Callable[..., Awaitable[Any]] = completion or litellm.acompletion
    request = {**openai_request, "stream": True}
    log.info("dispatch", shape="openai", model=request.get("model"), stream=True)
    stream = await dispatch(**request)
    async for chunk in stream:
        yield _sse_event(json.dumps(_coerce_to_dict(chunk)))
    yield _sse_event("[DONE]")
