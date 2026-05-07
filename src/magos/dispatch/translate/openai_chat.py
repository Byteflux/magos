"""`/v1/chat/completions` translate path via `litellm.acompletion`."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import litellm

from magos.dispatch.translate.payload import coerce_to_dict
from magos.dispatch.translate.runner import TranslateAdapter
from magos.dispatch.translate.sse import sse_event
from magos.shapes import OPENAI_CHAT


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
    shape=OPENAI_CHAT,
    endpoint="/v1/chat/completions",
    default_dispatch=litellm.acompletion,
    set_model_in_response=_chat_set_model_in_response,
    set_model_in_stream_event=_chat_set_model_in_stream_event,
    stream_bytes_iter=_openai_chat_bytes_iter,
    log_shape="openai",
)
