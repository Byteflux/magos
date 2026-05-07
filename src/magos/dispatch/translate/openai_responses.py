"""`/v1/responses` translate path via `litellm.aresponses`.

Streaming uses named SSE events (`event:` + `data:` per chunk).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import litellm

from magos.dispatch.translate.payload import coerce_to_dict
from magos.dispatch.translate.runner import TranslateAdapter
from magos.dispatch.translate.sse import sse_named_event
from magos.shapes import OPENAI_RESPONSES


def _responses_set_model_in_response(body: dict[str, Any], client_model: str) -> None:
    if "model" in body:
        body["model"] = client_model
    elif isinstance(body.get("response"), dict) and "model" in body["response"]:
        body["response"]["model"] = client_model


def _responses_set_model_in_stream_event(
    client_model: str,
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
    shape=OPENAI_RESPONSES,
    endpoint="/v1/responses",
    default_dispatch=litellm.aresponses,
    set_model_in_response=_responses_set_model_in_response,
    set_model_in_stream_event=_responses_set_model_in_stream_event,
    stream_bytes_iter=_openai_responses_bytes_iter,
    log_shape="openai-responses",
)
