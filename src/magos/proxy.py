"""Translate-mode dispatch into LiteLLM SDK call sites.

Three endpoint families, each backed by the LiteLLM SDK call that natively
handles its wire shape:

- ``/v1/messages``         -> ``litellm.anthropic_messages`` (Anthropic shape in,
                             Anthropic shape out, including cross-provider routing)
- ``/v1/chat/completions`` -> ``litellm.acompletion``        (OpenAI Chat Completions)
- ``/v1/responses``        -> ``litellm.aresponses``         (OpenAI Responses)

Each entry point accepts a ``completion`` callable for tests; production
wires the matching SDK function. Anthropic streaming returns raw SSE bytes
from LiteLLM, forwarded verbatim. OpenAI streaming wraps chunks into SSE
frames here because the SDK yields parsed objects.

Caller (``magos.routing.dispatch``) supplies ``dispatch_model`` already in
the form LiteLLM expects (``<provider>/<name>`` for unprefixed inputs); this
module no longer infers a provider from the model name.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol

import litellm

from magos.obs import get_logger, traced

log = get_logger("magos.proxy")

# Headers that the upstream HTTP client (litellm/openai-sdk/httpx) generates
# from the serialized request body. Forwarding the inbound values into
# ``extra_headers`` conflicts with that machinery: e.g. an inbound
# ``content-type: application/json`` overrides the SDK's own header and the
# upstream sees a body it cannot parse, returning "you must provide a model
# parameter". Server-level blocking (server._BLOCKED_FORWARD_HEADERS) is for
# the byte-exact passthrough path which legitimately needs ``content-type``.
_DISPATCH_BLOCKED_HEADERS: frozenset[str] = frozenset(
    {"content-type", "content-length", "content-encoding", "accept-encoding"}
)


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
    """OpenAI Responses streaming uses ``event:`` + ``data:`` lines per chunk."""
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


def _build_payload(
    request: dict[str, Any],
    *,
    dispatch_model: str,
    forward_headers: dict[str, str] | None,
    api_key: str | None,
    api_base: str | None = None,
) -> dict[str, Any]:
    """Compose the kwargs handed to a LiteLLM SDK call.

    ``dispatch_model`` overrides ``request["model"]`` because the routing
    layer has already chosen the LiteLLM-prefixed identifier; the inbound
    body's model may be a bare alias the operator declared.
    ``forward_headers`` are merged into ``extra_headers`` so upstream sees
    client auth, version pins, and beta flags verbatim, preserving the
    provider's billing shape. ``api_key`` is forwarded to LiteLLM when set
    so a rule's ``api_key_env`` can route across multiple keys per provider.
    ``api_base`` overrides LiteLLM's per-provider default URL; required for
    openai-compatible third parties (e.g. Vultr) routed through the generic
    ``custom_openai`` provider, where LiteLLM has no built-in host to fall
    back on.
    """
    out = dict(request)
    out["model"] = dispatch_model
    if forward_headers:
        safe = {
            k: v for k, v in forward_headers.items() if k.lower() not in _DISPATCH_BLOCKED_HEADERS
        }
        if safe:
            existing = out.get("extra_headers") or {}
            out["extra_headers"] = {**existing, **safe}
    if api_key is not None:
        out["api_key"] = api_key
    if api_base is not None:
        out["api_base"] = api_base
    return out


@traced("proxy.anthropic_messages")
async def proxy_anthropic_messages(
    anthropic_request: dict[str, Any],
    *,
    dispatch_model: str,
    completion: _CompletionFn | None = None,
    forward_headers: dict[str, str] | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> dict[str, Any]:
    """Round-trip an Anthropic Messages request through ``litellm.anthropic_messages``.

    LiteLLM's Anthropic-unified endpoint accepts Anthropic-shape input and
    emits Anthropic-shape output regardless of upstream provider, so this
    is a thin marshalling layer with no translation work of its own.
    """
    dispatch: Callable[..., Awaitable[Any]] = completion or litellm.anthropic_messages
    payload = _build_payload(
        anthropic_request,
        dispatch_model=dispatch_model,
        forward_headers=forward_headers,
        api_key=api_key,
        api_base=api_base,
    )
    log.info("dispatch", shape="anthropic", model=dispatch_model)
    return _coerce_to_dict(await dispatch(**payload))


def stream_anthropic_messages(
    anthropic_request: dict[str, Any],
    *,
    dispatch_model: str,
    completion: _CompletionFn | None = None,
    forward_headers: dict[str, str] | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> AsyncIterator[bytes]:
    """Stream an Anthropic Messages request via ``litellm.anthropic_messages``.

    Returned as a regular function (not an async generator) so request
    validation runs synchronously: a malformed request raises
    ``pydantic.ValidationError`` (inside LiteLLM) before any response bytes
    are emitted, and the server can surface it as 400 rather than mid-stream.

    LiteLLM yields raw Anthropic SSE bytes (``event: message_start``,
    ``content_block_delta``, etc.); we forward them verbatim.
    """
    dispatch: Callable[..., Awaitable[Any]] = completion or litellm.anthropic_messages
    payload = _build_payload(
        {**anthropic_request, "stream": True},
        dispatch_model=dispatch_model,
        forward_headers=forward_headers,
        api_key=api_key,
        api_base=api_base,
    )
    log.info("dispatch", shape="anthropic", model=dispatch_model, stream=True)
    return _anthropic_bytes_iter(payload, dispatch)


async def _anthropic_bytes_iter(
    payload: dict[str, Any],
    dispatch: Callable[..., Awaitable[Any]],
) -> AsyncIterator[bytes]:
    try:
        stream = await dispatch(**payload)
        async for chunk in stream:
            # LiteLLM 1.82+ yields bytes already SSE-framed for the
            # Anthropic-unified path; coerce the rare str chunk for safety.
            yield chunk if isinstance(chunk, bytes) else str(chunk).encode()
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


@traced("proxy.openai_chat_completions")
async def proxy_openai_chat_completions(
    openai_request: dict[str, Any],
    *,
    dispatch_model: str,
    completion: _CompletionFn | None = None,
    forward_headers: dict[str, str] | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> dict[str, Any]:
    """Pass an OpenAI Chat Completions request through litellm without translation."""
    dispatch: Callable[..., Awaitable[Any]] = completion or litellm.acompletion
    payload = _build_payload(
        openai_request,
        dispatch_model=dispatch_model,
        forward_headers=forward_headers,
        api_key=api_key,
        api_base=api_base,
    )
    log.info("dispatch", shape="openai", model=dispatch_model)
    return _coerce_to_dict(await dispatch(**payload))


async def stream_openai_chat_completions(
    openai_request: dict[str, Any],
    *,
    dispatch_model: str,
    completion: _CompletionFn | None = None,
    forward_headers: dict[str, str] | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> AsyncIterator[bytes]:
    """Stream OpenAI Chat Completions chunks as SSE bytes.

    Forces ``stream=True`` on the upstream call. Each chunk is JSON-encoded into
    a ``data: ...`` SSE event; the stream terminates with ``data: [DONE]``,
    matching OpenAI's wire format so existing OpenAI clients work unchanged.
    """
    dispatch: Callable[..., Awaitable[Any]] = completion or litellm.acompletion
    request = _build_payload(
        {**openai_request, "stream": True},
        dispatch_model=dispatch_model,
        forward_headers=forward_headers,
        api_key=api_key,
        api_base=api_base,
    )
    log.info("dispatch", shape="openai", model=dispatch_model, stream=True)
    stream = await dispatch(**request)
    async for chunk in stream:
        yield _sse_event(json.dumps(_coerce_to_dict(chunk)))
    yield _sse_event("[DONE]")


@traced("proxy.openai_responses")
async def proxy_openai_responses(
    openai_request: dict[str, Any],
    *,
    dispatch_model: str,
    completion: _CompletionFn | None = None,
    forward_headers: dict[str, str] | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> dict[str, Any]:
    """Pass an OpenAI Responses request through litellm without translation."""
    dispatch: Callable[..., Awaitable[Any]] = completion or litellm.aresponses
    payload = _build_payload(
        openai_request,
        dispatch_model=dispatch_model,
        forward_headers=forward_headers,
        api_key=api_key,
        api_base=api_base,
    )
    log.info("dispatch", shape="openai-responses", model=dispatch_model)
    return _coerce_to_dict(await dispatch(**payload))


async def stream_openai_responses(
    openai_request: dict[str, Any],
    *,
    dispatch_model: str,
    completion: _CompletionFn | None = None,
    forward_headers: dict[str, str] | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> AsyncIterator[bytes]:
    """Stream OpenAI Responses events as SSE bytes.

    Forces ``stream=True`` on the upstream call. Each event is JSON-encoded
    into an ``event: <type>\\ndata: <json>\\n\\n`` SSE frame, matching
    OpenAI's wire format so existing Responses clients work unchanged.
    """
    dispatch: Callable[..., Awaitable[Any]] = completion or litellm.aresponses
    request = _build_payload(
        {**openai_request, "stream": True},
        dispatch_model=dispatch_model,
        forward_headers=forward_headers,
        api_key=api_key,
        api_base=api_base,
    )
    log.info("dispatch", shape="openai-responses", model=dispatch_model, stream=True)
    stream = await dispatch(**request)
    async for chunk in stream:
        event = _coerce_to_dict(chunk)
        yield _sse_named_event(event)
