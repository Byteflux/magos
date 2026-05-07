"""``TranslateAdapter`` wiring for ``/v1/messages``.

Holds the response/stream model rewriters, the bytes iterator (which
also emits an Anthropic-shape error event when the upstream dispatch
raises), and the assembled ``ADAPTER`` consumed by the runner.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from magos.egress.translate.runner import TranslateAdapter
from magos.egress.translate.sse import sse_named_event
from magos.telemetry import get_logger

from .dispatch import _dispatch_anthropic_messages
from .translation import strip_anthropic_extras

log = get_logger("magos.egress.translate")


def _set_model_in_response(body: dict[str, Any], client_model: str) -> None:
    body["model"] = client_model


def _set_model_in_stream_event(
    client_model: str,
) -> Callable[[dict[str, Any]], bool]:
    def _mutate(data: dict[str, Any]) -> bool:
        msg = data.get("message")
        if isinstance(msg, dict) and "model" in msg:
            msg["model"] = client_model
            return True
        if "model" in data:
            data["model"] = client_model
            return True
        return False

    return _mutate


async def _bytes_iter(
    payload: dict[str, Any],
    dispatch: Callable[..., Awaitable[Any]],
) -> AsyncIterator[bytes]:
    try:
        stream = await dispatch(**payload)
        async for chunk in stream:
            # LiteLLM 1.82+ yields bytes already SSE-framed for the
            # Anthropic-unified path; coerce the rare str chunk for safety.
            yield chunk if isinstance(chunk, bytes) else str(chunk).encode()
    except Exception as exc:
        log.error(
            "stream.dispatch_failed",
            error=str(exc),
            error_type=type(exc).__name__,
            model=payload.get("model"),
        )
        # Emit an Anthropic-shape error event so the client can surface the
        # failure cleanly instead of seeing a truncated stream and retrying.
        yield sse_named_event(
            {
                "type": "error",
                "error": {"type": "api_error", "message": f"{type(exc).__name__}: {exc}"},
            }
        )


def _preprocess_body(
    body: dict[str, Any], dispatch_model: str, client_model: str
) -> dict[str, Any]:
    return strip_anthropic_extras(body, dispatch_model, client_model=client_model)


ADAPTER = TranslateAdapter(
    shape="anthropic",
    endpoint="/v1/messages",
    default_dispatch=_dispatch_anthropic_messages,
    set_model_in_response=_set_model_in_response,
    set_model_in_stream_event=_set_model_in_stream_event,
    stream_bytes_iter=_bytes_iter,
    log_shape="anthropic",
    preprocess_body=_preprocess_body,
)
