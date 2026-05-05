"""``/v1/responses`` translate path via ``litellm.aresponses``.

Streaming uses named SSE events (``event:`` + ``data:`` per chunk).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import litellm

from magos.egress.translate.payload import CompletionFn, coerce_to_dict
from magos.egress.translate.runner import TranslateAdapter, proxy_translate, stream_translate
from magos.egress.translate.sse import sse_named_event


def _responses_set_model_in_response(body: dict[str, Any], client_model: str) -> None:
    if "model" in body:
        body["model"] = client_model
    elif isinstance(body.get("response"), dict) and "model" in body["response"]:
        body["response"]["model"] = client_model


def _responses_set_model_in_stream_event(
    _payload: dict[str, Any], client_model: str
) -> Callable[[dict[str, Any]], bool]:
    def _mutate(data: dict[str, Any]) -> bool:
        if "model" in data:
            data["model"] = client_model
            return True
        nested = data.get("response")
        if isinstance(nested, dict) and "model" in nested:
            nested["model"] = client_model
            return True
        return False

    return _mutate


async def _openai_responses_bytes_iter(
    payload: dict[str, Any],
    dispatch: Callable[..., Awaitable[Any]],
) -> AsyncIterator[bytes]:
    stream = await dispatch(**payload)
    async for chunk in stream:
        event = coerce_to_dict(chunk)
        yield sse_named_event(event)


ADAPTER = TranslateAdapter(
    shape="openai-responses",
    endpoint="/v1/responses",
    default_dispatch=litellm.aresponses,
    set_model_in_response=_responses_set_model_in_response,
    set_model_in_stream_event=_responses_set_model_in_stream_event,
    stream_bytes_iter=_openai_responses_bytes_iter,
    traced_name="proxy.openai_responses",
    log_shape="openai-responses",
)


async def proxy_openai_responses(
    openai_request: dict[str, Any],
    *,
    dispatch_model: str,
    provider: str | None = None,
    completion: CompletionFn | None = None,
    forward_headers: dict[str, str] | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> dict[str, Any]:
    """Pass an OpenAI Responses request through litellm without translation."""
    return await proxy_translate(
        ADAPTER,
        openai_request,
        dispatch_model=dispatch_model,
        provider=provider,
        completion=completion,
        forward_headers=forward_headers,
        api_key=api_key,
        api_base=api_base,
    )


def stream_openai_responses(
    openai_request: dict[str, Any],
    *,
    dispatch_model: str,
    provider: str | None = None,
    completion: CompletionFn | None = None,
    forward_headers: dict[str, str] | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> AsyncIterator[bytes]:
    """Stream OpenAI Responses events as SSE bytes."""
    return stream_translate(
        ADAPTER,
        openai_request,
        dispatch_model=dispatch_model,
        provider=provider,
        completion=completion,
        forward_headers=forward_headers,
        api_key=api_key,
        api_base=api_base,
    )
