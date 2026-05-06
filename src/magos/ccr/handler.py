"""Magos-side wrappers around headroom's CCR response handlers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from headroom.ccr import CCRResponseHandler, ResponseHandlerConfig

from magos.routing.request import RoutedRequest
from magos.telemetry import get_logger

from .continuation import make_continuation_callable
from .detection import is_ccr_request

if TYPE_CHECKING:
    from magos.egress.translate import TranslateAdapter

log = get_logger("magos.ccr")


async def wrap_response(
    response: dict[str, Any],
    *,
    req: RoutedRequest,
    adapter: TranslateAdapter,
    completion: Callable[..., Awaitable[Any]],
    dispatch_model: str,
    provider: str,
    forward_headers: dict[str, str],
    api_key: str | None,
    api_base: str | None,
) -> dict[str, Any]:
    """Hand a non-streaming response through headroom's CCR handler.

    Short-circuits when the request didn't inject the CCR tool (cheapest
    fast path) or when the response contains no CCR tool calls. Otherwise
    constructs the continuation closure, instantiates a per-request
    ``CCRResponseHandler``, and returns the post-continuation response.
    """
    if not is_ccr_request(req):
        return response

    handler = CCRResponseHandler(ResponseHandlerConfig())
    if not handler.has_ccr_tool_calls(response, provider=provider):
        return response

    continuation = make_continuation_callable(
        adapter=adapter,
        original_body=dict(req.body),
        completion=completion,
        dispatch_model=dispatch_model,
        provider=provider,
        forward_headers=forward_headers,
        api_key=api_key,
        api_base=api_base,
    )

    messages = list(req.body.get("messages", []))
    tools = req.body.get("tools")
    log.info("ccr.wrap_response_start", endpoint=req.endpoint, provider=provider)
    try:
        final = await handler.handle_response(
            response, messages, tools, continuation, provider=provider
        )
    except Exception as exc:
        log.warning(
            "ccr.handler_failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return response
    log.info(
        "ccr.wrap_response_done",
        endpoint=req.endpoint,
        retrievals=handler.get_stats().get("total_retrievals", 0),
    )
    return final
