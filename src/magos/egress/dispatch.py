"""Bridge from a ``RouteDecision`` to the translate / passthrough seams.

The dispatcher is the only egress module that knows about FastAPI
response types. ``magos.ingress.http.run`` hands it a decision already
produced by ``route()``; the dispatcher picks the right underlying call
based on ``req.endpoint``, ``action.mode``, and the request's ``stream``
flag.

API-key resolution and per-provider auth-header injection live in
:mod:`magos.egress.auth`. ``DispatchError`` is re-exported from this
module because callers catch it at the dispatch boundary — keeping the
exception in one logical place with the function that raises it.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Response
from fastapi.responses import StreamingResponse

from magos.egress.auth import DispatchError, maybe_inject_api_key, resolve_api_key
from magos.egress.passthrough import _HTTP_ERROR_THRESHOLD, call_passthrough, stream_passthrough
from magos.egress.tokens import count_tokens
from magos.egress.translate import (
    proxy_anthropic_messages,
    proxy_openai_chat_completions,
    proxy_openai_responses,
    stream_anthropic_messages,
    stream_openai_chat_completions,
    stream_openai_responses,
)
from magos.egress.usage import (
    log_usage_from_body,
    shape_for_endpoint,
    tap_stream,
)
from magos.routing.engine import RouteDecision
from magos.telemetry import get_logger

__all__ = ["DispatchError", "dispatch_decision"]

log = get_logger("magos.egress.dispatch")

CompletionFn = Callable[..., Awaitable[Any]]


async def dispatch_decision(  # noqa: PLR0911, PLR0912
    decision: RouteDecision,
    *,
    completion: CompletionFn,
) -> Response | StreamingResponse | dict[str, Any]:
    """Hand ``decision`` off to the right downstream call site.

    Branches: count_tokens, passthrough+stream, passthrough+non-stream,
    translate x {messages, chat, responses} x {stream, non-stream}.
    """
    req = decision.request
    action = decision.action

    if req.endpoint == "/v1/messages/count_tokens":
        return await _dispatch_count_tokens(decision, completion=completion)

    forward_headers = maybe_inject_api_key(dict(req.headers), action)
    is_streaming = bool(req.body.get("stream"))

    if action.mode == "passthrough":
        if not action.base_url:  # validated at config load; defensive guard.
            raise DispatchError("passthrough rule has no base_url")
        body_bytes = req.raw_body if not req.body_dirty else json.dumps(dict(req.body)).encode()
        model_hint = str(req.body.get("model", ""))
        shape = shape_for_endpoint(req.endpoint)
        if is_streaming:
            upstream = stream_passthrough(
                body_bytes,
                forward_headers,
                action.base_url,
                path=req.forward_path,
                method=req.method,
                model_hint=model_hint,
            )
            iterator = (
                tap_stream(
                    upstream, shape, endpoint=req.endpoint, fallback_model=model_hint or None
                )
                if shape is not None
                else upstream
            )
            return StreamingResponse(iterator, media_type="text/event-stream")
        status, raw, content_type = await call_passthrough(
            body_bytes,
            forward_headers,
            action.base_url,
            path=req.forward_path,
            method=req.method,
            model_hint=model_hint,
        )
        if (
            shape is not None
            and status < _HTTP_ERROR_THRESHOLD
            and content_type.startswith("application/json")
        ):
            try:
                parsed = json.loads(raw)
            except (UnicodeDecodeError, ValueError):
                parsed = None
            if parsed is not None:
                log_usage_from_body(shape, parsed, endpoint=req.endpoint)
        return Response(content=raw, status_code=status, media_type=content_type)

    # mode: translate -- only POST endpoints have litellm equivalents.
    if req.method != "POST":
        raise DispatchError(
            f"mode='translate' does not support method={req.method!r}; "
            "use mode='passthrough' for auxiliary GET/DELETE endpoints"
        )
    api_key = resolve_api_key(action.api_key_env)
    api_base = action.base_url

    if req.endpoint == "/v1/messages":
        if is_streaming:
            stream = stream_anthropic_messages(
                dict(req.body),
                dispatch_model=decision.dispatch_model,
                provider=action.provider,
                completion=completion,
                forward_headers=forward_headers,
                api_key=api_key,
                api_base=api_base,
            )
            return StreamingResponse(stream, media_type="text/event-stream")
        return await proxy_anthropic_messages(
            dict(req.body),
            dispatch_model=decision.dispatch_model,
            provider=action.provider,
            completion=completion,
            forward_headers=forward_headers,
            api_key=api_key,
            api_base=api_base,
        )

    if req.endpoint == "/v1/chat/completions":
        if is_streaming:
            stream = stream_openai_chat_completions(
                dict(req.body),
                dispatch_model=decision.dispatch_model,
                provider=action.provider,
                completion=completion,
                forward_headers=forward_headers,
                api_key=api_key,
                api_base=api_base,
            )
            return StreamingResponse(stream, media_type="text/event-stream")
        return await proxy_openai_chat_completions(
            dict(req.body),
            dispatch_model=decision.dispatch_model,
            provider=action.provider,
            completion=completion,
            forward_headers=forward_headers,
            api_key=api_key,
            api_base=api_base,
        )

    # /v1/responses
    if is_streaming:
        stream = stream_openai_responses(
            dict(req.body),
            dispatch_model=decision.dispatch_model,
            provider=action.provider,
            completion=completion,
            forward_headers=forward_headers,
            api_key=api_key,
            api_base=api_base,
        )
        return StreamingResponse(stream, media_type="text/event-stream")
    return await proxy_openai_responses(
        dict(req.body),
        dispatch_model=decision.dispatch_model,
        provider=action.provider,
        completion=completion,
        forward_headers=forward_headers,
        api_key=api_key,
        api_base=api_base,
    )


async def _dispatch_count_tokens(
    decision: RouteDecision, *, completion: CompletionFn
) -> dict[str, int]:
    """Dispatch a count_tokens request via ``litellm.acount_tokens``."""
    body = dict(decision.request.body)
    n = await count_tokens(
        body,
        dispatch_model=decision.dispatch_model,
        count=completion,
    )
    return {"input_tokens": n}
