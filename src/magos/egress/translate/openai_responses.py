"""``/v1/responses`` translate path via ``litellm.aresponses``.

OpenAI Responses in, OpenAI Responses out. Streaming uses named SSE
events (``event: <type>\\ndata: <json>\\n\\n``) per the Responses wire
format.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import litellm

from magos.egress.translate.payload import (
    CompletionFn,
    build_payload,
    coerce_to_dict,
)
from magos.egress.translate.sse import sse_named_event
from magos.egress.usage import log_usage_from_body, tap_stream
from magos.telemetry import get_logger, traced

log = get_logger("magos.egress.translate")


@traced("proxy.openai_responses")
async def proxy_openai_responses(
    openai_request: dict[str, Any],
    *,
    dispatch_model: str,
    completion: CompletionFn | None = None,
    forward_headers: dict[str, str] | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> dict[str, Any]:
    """Pass an OpenAI Responses request through litellm without translation."""
    dispatch: Callable[..., Awaitable[Any]] = completion or litellm.aresponses
    payload = build_payload(
        openai_request,
        dispatch_model=dispatch_model,
        forward_headers=forward_headers,
        api_key=api_key,
        api_base=api_base,
    )
    log.info("dispatch", shape="openai-responses", model=dispatch_model)
    body = coerce_to_dict(await dispatch(**payload))
    log_usage_from_body("openai-responses", body, endpoint="/v1/responses")
    return body


def stream_openai_responses(
    openai_request: dict[str, Any],
    *,
    dispatch_model: str,
    completion: CompletionFn | None = None,
    forward_headers: dict[str, str] | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> AsyncIterator[bytes]:
    """Stream OpenAI Responses events as SSE bytes."""
    dispatch: Callable[..., Awaitable[Any]] = completion or litellm.aresponses
    request = build_payload(
        {**openai_request, "stream": True},
        dispatch_model=dispatch_model,
        forward_headers=forward_headers,
        api_key=api_key,
        api_base=api_base,
    )
    log.info("dispatch", shape="openai-responses", model=dispatch_model, stream=True)
    return tap_stream(
        _openai_responses_bytes_iter(request, dispatch),
        "openai-responses",
        endpoint="/v1/responses",
        fallback_model=dispatch_model,
    )


async def _openai_responses_bytes_iter(
    request: dict[str, Any],
    dispatch: Callable[..., Awaitable[Any]],
) -> AsyncIterator[bytes]:
    stream = await dispatch(**request)
    async for chunk in stream:
        event = coerce_to_dict(chunk)
        yield sse_named_event(event)
