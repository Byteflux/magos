"""``/v1/chat/completions`` translate path via ``litellm.acompletion``.

OpenAI Chat Completions in, OpenAI Chat Completions out. LiteLLM
handles the per-provider translation when the dispatch model points
elsewhere.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import litellm

from magos.egress.translate.payload import (
    CompletionFn,
    build_payload,
    coerce_to_dict,
    resolve_client_model,
)
from magos.egress.translate.sse import rewrite_data_in_stream, sse_event
from magos.egress.usage import log_usage_from_body, tap_stream
from magos.telemetry import get_logger, traced

log = get_logger("magos.egress.translate")


@traced("proxy.openai_chat_completions")
async def proxy_openai_chat_completions(
    openai_request: dict[str, Any],
    *,
    dispatch_model: str,
    provider: str | None = None,
    completion: CompletionFn | None = None,
    forward_headers: dict[str, str] | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> dict[str, Any]:
    """Pass an OpenAI Chat Completions request through litellm without translation."""
    dispatch: Callable[..., Awaitable[Any]] = completion or litellm.acompletion
    payload = build_payload(
        openai_request,
        dispatch_model=dispatch_model,
        forward_headers=forward_headers,
        api_key=api_key,
        api_base=api_base,
    )
    client_model = resolve_client_model(openai_request.get("model", ""), provider, dispatch_model)
    log.info("dispatch", shape="openai", model=client_model, dispatch_model=dispatch_model)
    body = coerce_to_dict(await dispatch(**payload))
    body["model"] = client_model
    log_usage_from_body("openai-chat", body, endpoint="/v1/chat/completions")
    return body


def stream_openai_chat_completions(
    openai_request: dict[str, Any],
    *,
    dispatch_model: str,
    provider: str | None = None,
    completion: CompletionFn | None = None,
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
    request = build_payload(
        {**openai_request, "stream": True},
        dispatch_model=dispatch_model,
        forward_headers=forward_headers,
        api_key=api_key,
        api_base=api_base,
    )
    client_model = resolve_client_model(openai_request.get("model", ""), provider, dispatch_model)
    log.info(
        "dispatch",
        shape="openai",
        model=client_model,
        dispatch_model=dispatch_model,
        stream=True,
    )

    def _set_model(data: dict[str, Any]) -> bool:
        if "model" in data:
            data["model"] = client_model
            return True
        return False

    return tap_stream(
        rewrite_data_in_stream(_openai_chat_bytes_iter(request, dispatch), _set_model),
        "openai-chat",
        endpoint="/v1/chat/completions",
        fallback_model=client_model,
    )


async def _openai_chat_bytes_iter(
    request: dict[str, Any],
    dispatch: Callable[..., Awaitable[Any]],
) -> AsyncIterator[bytes]:
    stream = await dispatch(**request)
    async for chunk in stream:
        yield sse_event(json.dumps(coerce_to_dict(chunk)))
    yield sse_event("[DONE]")
