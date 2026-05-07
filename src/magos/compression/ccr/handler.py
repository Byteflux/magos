"""Magos-side wrappers around headroom's CCR response handlers."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

from headroom.ccr import CCR_TOOL_NAME, CCRResponseHandler, ResponseHandlerConfig

from magos.compression.ccr.continuation import make_continuation_callable
from magos.routing.request import RoutedRequest
from magos.telemetry import get_logger


def is_ccr_request(req: RoutedRequest) -> bool:
    """True when ``req.body['tools']`` contains the ``headroom_retrieve`` tool.

    Recognises both Anthropic shape (top-level ``name``) and OpenAI shape
    (``function.name``). Returns False for missing / empty / malformed tools.
    The compress rewrite is the only place that injects this tool, so
    presence is a self-describing signal that CCR is active for this
    request — no per-request side channel needed.
    """
    tools = req.body.get("tools")
    if not isinstance(tools, list):
        return False
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        # Anthropic shape
        if tool.get("name") == CCR_TOOL_NAME:
            return True
        # OpenAI shape
        function = tool.get("function")
        if isinstance(function, dict) and function.get("name") == CCR_TOOL_NAME:
            return True
    return False


if TYPE_CHECKING:
    from magos.dispatch.translate import TranslateAdapter

log = get_logger("magos.compression.ccr")


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


async def wrap_stream(
    upstream: AsyncIterator[bytes],
    *,
    req: RoutedRequest,
    adapter: TranslateAdapter,
    completion: Callable[..., Awaitable[Any]],
    dispatch_model: str,
    provider: str,
    forward_headers: dict[str, str],
    api_key: str | None,
    api_base: str | None,
) -> AsyncIterator[bytes]:
    """Hand a streaming response through headroom's CCR streaming handler.

    Short-circuits when the request didn't inject the CCR tool: passes
    chunks through verbatim. Otherwise wraps with
    ``StreamingCCRHandler.process_stream`` and forwards the chunks it
    yields (which may be the original stream or a continuation stream).
    """
    if not is_ccr_request(req):
        async for chunk in upstream:
            yield chunk
        return

    from headroom.ccr import StreamingCCRHandler  # noqa: PLC0415

    handler = StreamingCCRHandler(
        CCRResponseHandler(ResponseHandlerConfig()),
        provider=provider,
    )
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
    log.info("ccr.wrap_stream_start", endpoint=req.endpoint, provider=provider)
    try:
        async for chunk in handler.process_stream(upstream, messages, tools, continuation):
            yield chunk
    except Exception as exc:
        log.warning(
            "ccr.stream_handler_failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return
    log.info("ccr.wrap_stream_done", endpoint=req.endpoint)
