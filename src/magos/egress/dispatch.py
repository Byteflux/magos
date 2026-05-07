"""Branch a ``RouteDecision`` into translate / passthrough / count_tokens.

See ``docs/architecture/request-flow.md``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from fastapi import Response
from fastapi.responses import StreamingResponse

from magos.ccr import wrap_response, wrap_stream
from magos.egress import CompletionFn
from magos.egress.auth import maybe_inject_api_key, resolve_api_key
from magos.egress.errors import DispatchError
from magos.egress.passthrough import _HTTP_ERROR_THRESHOLD, call_passthrough, stream_passthrough
from magos.egress.tokens import count_tokens
from magos.egress.translate import TRANSLATE_HANDLERS
from magos.egress.translate.runner import proxy_translate, stream_translate
from magos.egress.usage import (
    Usage,
    log_usage_from_body,
    tap_stream,
)
from magos.routing import RouteDecision
from magos.routing.request import PostResponseHook
from magos.shapes import shape_for_endpoint
from magos.telemetry import get_logger

__all__ = ["dispatch_decision"]

log = get_logger("magos.egress.dispatch")


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
    target = decision.target

    if req.endpoint == "/v1/messages/count_tokens":
        n = await count_tokens(
            dict(req.body),
            dispatch_model=decision.dispatch_model,
            count=completion,
        )
        return {"input_tokens": n}

    forward_headers = maybe_inject_api_key(dict(req.headers), target)
    is_streaming = bool(req.body.get("stream"))

    on_complete = _make_on_complete(req.post_response_hooks)

    if target.gateway == "passthrough":
        if not target.base_url:  # validated at config load; defensive guard.
            raise DispatchError("passthrough rule has no base_url")
        body_bytes = req.raw_body if not req.body_dirty else json.dumps(dict(req.body)).encode()
        model_hint = str(req.body.get("model", ""))
        shape = shape_for_endpoint(req.endpoint)
        if is_streaming:
            upstream = stream_passthrough(
                body_bytes,
                forward_headers,
                target.base_url,
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
            target.base_url,
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

    # gateway: translate -- only POST endpoints have litellm equivalents.
    if req.method != "POST":
        raise DispatchError(
            f"gateway='translate' does not support method={req.method!r}; "
            "use gateway='passthrough' for auxiliary GET/DELETE endpoints"
        )
    api_key = resolve_api_key(target.api_key_env)
    api_base = target.base_url

    adapter = TRANSLATE_HANDLERS.get(req.endpoint)
    if adapter is None:
        raise DispatchError(f"no translate handler for endpoint {req.endpoint!r}")

    # Shared dispatch parameters; ``proxy_translate``/``stream_translate``
    # also need ``on_complete``, ``wrap_response``/``wrap_stream`` also need
    # ``req`` and ``adapter``. Building from one base dict keeps the two
    # call sites in lock-step when a parameter is added.
    shared: dict[str, Any] = {
        "dispatch_model": decision.dispatch_model,
        "provider": target.provider,
        "completion": completion,
        "forward_headers": forward_headers,
        "api_key": api_key,
        "api_base": api_base,
    }
    translate_kwargs = {**shared, "on_complete": on_complete}
    ccr_kwargs = {**shared, "req": req, "adapter": adapter}

    if is_streaming:
        stream = stream_translate(adapter, dict(req.body), **translate_kwargs)
        return StreamingResponse(
            wrap_stream(stream, **ccr_kwargs),
            media_type="text/event-stream",
        )
    response = await proxy_translate(adapter, dict(req.body), **translate_kwargs)
    return await wrap_response(response, **ccr_kwargs)
