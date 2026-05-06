"""Branch a ``RouteDecision`` into translate / passthrough / count_tokens.

See ``docs/architecture/request-flow.md``.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Response
from fastapi.responses import StreamingResponse

from magos.egress.auth import maybe_inject_api_key, resolve_api_key
from magos.egress.errors import DispatchError
from magos.egress.passthrough import _HTTP_ERROR_THRESHOLD, call_passthrough, stream_passthrough
from magos.egress.tokens import count_tokens
from magos.egress.translate import TRANSLATE_HANDLERS
from magos.egress.translate.runner import proxy_translate, stream_translate
from magos.egress.usage import (
    Usage,
    log_usage_from_body,
    shape_for_endpoint,
    tap_stream,
)
from magos.routing import RouteDecision
from magos.routing.request import PostResponseHook
from magos.telemetry import get_logger

__all__ = ["dispatch_decision"]

log = get_logger("magos.egress.dispatch")

CompletionFn = Callable[..., Awaitable[Any]]


def _make_on_complete(
    hooks: list[PostResponseHook],
) -> Callable[[Usage], None] | None:
    """Wrap a hook list into a single on_complete callback.

    Returns None when there are no hooks (so call sites can pass it
    through unconditionally without paying the wrap cost). Each hook
    is fired in order; one raising hook is logged and skipped, the
    rest still fire.
    """
    if not hooks:
        return None
    snapshot = list(hooks)

    def fire(usage: Usage) -> None:
        for hook in snapshot:
            try:
                hook(usage)
            except Exception as exc:
                log.warning(
                    "compress.hook_failed",
                    hook=getattr(hook, "__qualname__", repr(hook)),
                    error=str(exc),
                    error_type=type(exc).__name__,
                )

    return fire


async def dispatch_decision(
    decision: RouteDecision,
    *,
    completion: CompletionFn,
) -> Response | StreamingResponse | dict[str, Any]:
    """Hand ``decision`` off to the right downstream call site."""
    req = decision.request
    action = decision.action

    if req.endpoint == "/v1/messages/count_tokens":
        return await _dispatch_count_tokens(decision, completion=completion)

    forward_headers = maybe_inject_api_key(dict(req.headers), action)
    is_streaming = bool(req.body.get("stream"))

    on_complete = _make_on_complete(req.post_response_hooks)

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
                    upstream,
                    shape,
                    endpoint=req.endpoint,
                    fallback_model=model_hint or None,
                    on_complete=on_complete,
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
                log_usage_from_body(shape, parsed, endpoint=req.endpoint, on_complete=on_complete)
        return Response(content=raw, status_code=status, media_type=content_type)

    # mode: translate -- only POST endpoints have litellm equivalents.
    if req.method != "POST":
        raise DispatchError(
            f"mode='translate' does not support method={req.method!r}; "
            "use mode='passthrough' for auxiliary GET/DELETE endpoints"
        )
    api_key = resolve_api_key(action.api_key_env)
    api_base = action.base_url

    adapter = TRANSLATE_HANDLERS.get(req.endpoint)
    if adapter is None:
        raise DispatchError(f"no translate handler for endpoint {req.endpoint!r}")

    common: dict[str, Any] = {
        "dispatch_model": decision.dispatch_model,
        "provider": action.provider,
        "completion": completion,
        "forward_headers": forward_headers,
        "api_key": api_key,
        "api_base": api_base,
        "on_complete": on_complete,
    }
    if is_streaming:
        stream = stream_translate(adapter, dict(req.body), **common)
        return StreamingResponse(stream, media_type="text/event-stream")
    return await proxy_translate(adapter, dict(req.body), **common)


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
