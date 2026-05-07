"""``/v1/chat/completions`` translate path via ``litellm.acompletion``."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import litellm

from magos.egress import CompletionFn
from magos.egress.translate.payload import coerce_to_dict
from magos.egress.translate.runner import TranslateAdapter, proxy_translate, stream_translate
from magos.egress.translate.sse import sse_event


def _chat_set_model_in_response(body: dict[str, Any], client_model: str) -> None:
    body["model"] = client_model


def _chat_set_model_in_stream_event(
    client_model: str,
) -> Callable[[dict[str, Any]], bool]:
    def _mutate(data: dict[str, Any]) -> bool:
        if "model" in data:
            data["model"] = client_model
            return True
        return False

    return _mutate


async def _openai_chat_bytes_iter(
    payload: dict[str, Any],
    dispatch: Callable[..., Awaitable[Any]],
) -> AsyncIterator[bytes]:
    stream = await dispatch(**payload)
    async for chunk in stream:
        yield sse_event(json.dumps(coerce_to_dict(chunk)))
    yield sse_event("[DONE]")


ADAPTER = TranslateAdapter(
    shape="openai-chat",
    endpoint="/v1/chat/completions",
    default_dispatch=litellm.acompletion,
    set_model_in_response=_chat_set_model_in_response,
    set_model_in_stream_event=_chat_set_model_in_stream_event,
    stream_bytes_iter=_openai_chat_bytes_iter,
    log_shape="openai",
)


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
    """Stream OpenAI Chat Completions chunks as SSE bytes terminated by ``[DONE]``."""
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
