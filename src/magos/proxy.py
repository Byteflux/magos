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

# Cross-shape translation routinely surfaces params one provider supports and
# another does not (e.g. Anthropic's ``context_management`` arriving on a
# request routed to ``custom_openai`` for Vultr). LiteLLM's per-provider
# allow-lists raise ``UnsupportedParamsError`` by default; flipping
# ``drop_params`` to True makes it silently drop unsupported params at the
# destination only — supported providers (Anthropic for ``context_management``)
# still receive them. Without this, every new client-side feature that lands
# in Claude Code or the OpenAI SDK breaks routing to alt providers until we
# patch a request rewrite.
litellm.drop_params = True

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

# Auth headers describing the inbound (client -> magos) hop. When the
# operator has chosen an upstream key explicitly via ``api_key``, these
# must NOT be forwarded into ``extra_headers``: the openai-sdk lets
# ``extra_headers`` override the ``api_key`` kwarg, so leaking the
# inbound bearer to a different upstream provider produces a misleading
# "Invalid API key" 401 even though magos was invoked with a valid key.
# When ``api_key`` is None (rule has no ``api_key_env``), these stay in
# place so litellm's per-provider env-var resolution still wins.
_INBOUND_AUTH_HEADERS: frozenset[str] = frozenset({"authorization", "x-api-key"})

# Canonical Anthropic Messages body fields that LiteLLM's
# ``anthropic_messages`` translator knows how to map to non-Anthropic
# providers. Anything outside this set (e.g. ``context_management``,
# ``output_config``, future Anthropic-only additions) falls into ``**kwargs``
# and leaks straight into the OpenAI SDK on cross-provider routes, producing
# ``unexpected keyword argument`` errors. ``litellm.drop_params=True`` only
# helps for params LiteLLM *recognizes* but the destination doesn't support;
# unknown fields slip past it. We pre-filter the body when dispatching to a
# non-Anthropic upstream so new client-side features can't break routing.
_ANTHROPIC_MESSAGES_CANONICAL_FIELDS: frozenset[str] = frozenset(
    {
        "model",
        "messages",
        "max_tokens",
        "system",
        "temperature",
        "top_p",
        "top_k",
        "stop_sequences",
        "stream",
        "metadata",
        "tools",
        "tool_choice",
        "thinking",
        # OpenAI-shape fields produced by ``_translate_output_config``. They
        # ride through LiteLLM's ``anthropic_messages`` ``**kwargs`` to the
        # destination translator, which knows them.
        "reasoning_effort",
        "response_format",
    }
)

# Anthropic's ``output_config.effort`` accepts levels OpenAI's
# ``reasoning_effort`` does not (``xhigh`` and ``max``); clamp the high end so
# the destination doesn't reject the value. ``minimal`` is OpenAI-only and not
# emitted by Anthropic, so no inbound mapping is needed.
_ANTHROPIC_EFFORT_TO_OPENAI: dict[str, str] = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}


def _translate_output_config(body: dict[str, Any]) -> dict[str, Any]:
    """Map Anthropic ``output_config`` onto OpenAI-shape equivalents.

    Anthropic's ``output_config`` carries two operationally important fields:

    - ``effort`` -> OpenAI ``reasoning_effort`` (``xhigh``/``max`` clamp to ``high``)
    - ``format`` (json_schema) -> OpenAI ``response_format``

    LiteLLM's ``anthropic_messages`` translator predates ``output_config`` and
    forwards it as an unrecognized kwarg, which the destination SDK rejects.
    Dropping it loses structured-output and reasoning-effort behavior on
    cross-provider routes, so translate before dispatch. Caller-supplied
    ``reasoning_effort`` / ``response_format`` win over the derived values.
    """
    cfg = body.get("output_config")
    if not isinstance(cfg, dict):
        return body
    out = {k: v for k, v in body.items() if k != "output_config"}
    effort = cfg.get("effort")
    if isinstance(effort, str) and "reasoning_effort" not in out:
        mapped = _ANTHROPIC_EFFORT_TO_OPENAI.get(effort)
        if mapped is not None:
            out["reasoning_effort"] = mapped
    fmt = cfg.get("format")
    if isinstance(fmt, dict) and fmt.get("type") == "json_schema" and "response_format" not in out:
        # Anthropic nests the schema directly under ``format``; OpenAI wraps
        # it in a ``json_schema`` object. The schema body itself is identical.
        schema = fmt.get("schema")
        if isinstance(schema, dict):
            out["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": fmt.get("name", "response"),
                    "schema": schema,
                    **({"strict": True} if fmt.get("strict") else {}),
                },
            }
    return out


def _strip_anthropic_extras(body: dict[str, Any], dispatch_model: str) -> dict[str, Any]:
    """Drop Anthropic-only body fields when dispatching to a non-Anthropic upstream.

    ``litellm.anthropic_messages`` forwards Anthropic-shape requests verbatim
    to Anthropic and translates them to OpenAI/Bedrock/etc. for everything
    else. The translator only handles canonical fields; unknown extras flow
    through ``**kwargs`` and surface as ``unexpected keyword argument`` errors
    inside the destination SDK. ``output_config`` is translated to OpenAI
    equivalents first so structured-output / reasoning-effort behavior carries
    over; remaining Anthropic-only fields (``context_management``, future
    additions) are dropped. Anthropic-bound traffic is left untouched so
    everything passes through verbatim.
    """
    if dispatch_model.startswith("anthropic/"):
        return body
    body = _translate_output_config(body)
    extras = set(body) - _ANTHROPIC_MESSAGES_CANONICAL_FIELDS
    if not extras:
        return body
    log.info(
        "anthropic.dropped_unknown_fields",
        model=dispatch_model,
        fields=sorted(extras),
    )
    return {k: v for k, v in body.items() if k in _ANTHROPIC_MESSAGES_CANONICAL_FIELDS}


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
        blocked = _DISPATCH_BLOCKED_HEADERS
        if api_key is not None:
            # Operator picked the upstream key; don't let the inbound auth
            # header (claude code's anthropic token, etc.) leak into
            # extra_headers and override it on the openai-sdk hop.
            blocked = blocked | _INBOUND_AUTH_HEADERS
        safe = {k: v for k, v in forward_headers.items() if k.lower() not in blocked}
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
        _strip_anthropic_extras(anthropic_request, dispatch_model),
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
        {**_strip_anthropic_extras(anthropic_request, dispatch_model), "stream": True},
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
