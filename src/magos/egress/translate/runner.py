"""Generic translate runner driven by per-shape ``TranslateAdapter`` declarations.

Each shape (Anthropic Messages, OpenAI Chat, OpenAI Responses) registers a
``TranslateAdapter`` frozen dataclass that captures the shape-specific
variation points. ``proxy_translate`` and ``stream_translate`` contain the
logic that was previously duplicated across the three endpoint modules. See
``docs/architecture/translation.md``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from magos.egress.translate.payload import (
    CompletionFn,
    build_payload,
    coerce_to_dict,
    resolve_client_model,
)
from magos.egress.translate.sse import rewrite_data_in_stream
from magos.egress.usage import Shape, Usage, log_usage_from_body, tap_stream
from magos.telemetry import get_logger

log = get_logger("magos.egress.translate")


@dataclass(frozen=True)
class TranslateAdapter:
    """Per-shape variation points for the generic translate runner.

    ``shape`` is the log/usage label (e.g. ``"anthropic"``).
    ``endpoint`` is the API path (e.g. ``"/v1/messages"``).
    ``default_dispatch`` is the LiteLLM SDK function to call when the caller
    supplies no ``completion`` override.
    ``set_model_in_response`` mutates a non-streaming response body in-place
    to rewrite the model field(s) back to the client-facing id.
    ``set_model_in_stream_event`` is the ``mutator`` passed to
    ``rewrite_data_in_stream``; returns ``True`` iff it modified ``data``.
    ``preprocess_body`` is an optional hook called before ``build_payload``
    to perform shape-specific body transforms (e.g. Anthropic extras stripping).
    ``stream_bytes_iter`` is the async generator that reads chunks from the
    LiteLLM stream and emits SSE-framed bytes. Shape-specific because Anthropic
    yields pre-framed bytes while OpenAI shapes need explicit framing.
    ``traced_name`` is the OTel span label for the non-streaming path.
    ``log_shape`` is the label passed to ``log.info("dispatch", shape=...)``.
    """

    shape: Shape
    endpoint: str
    default_dispatch: Callable[..., Awaitable[Any]]
    set_model_in_response: Callable[[dict[str, Any], str], None]
    set_model_in_stream_event: Callable[[dict[str, Any], str], Callable[[dict[str, Any]], bool]]
    stream_bytes_iter: Callable[
        [dict[str, Any], Callable[..., Awaitable[Any]]], AsyncIterator[bytes]
    ]
    traced_name: str
    log_shape: str
    preprocess_body: Callable[[dict[str, Any], str, str], dict[str, Any]] | None = None


async def _proxy_translate_inner(
    adapter: TranslateAdapter,
    request: dict[str, Any],
    *,
    dispatch: Callable[..., Awaitable[Any]],
    dispatch_model: str,
    provider: str | None,
    forward_headers: dict[str, str] | None,
    api_key: str | None,
    api_base: str | None,
    on_complete: Callable[[Usage], None] | None = None,
) -> dict[str, Any]:
    client_model = resolve_client_model(request.get("model", ""), provider, dispatch_model)
    body = request
    if adapter.preprocess_body is not None:
        body = adapter.preprocess_body(body, dispatch_model, client_model)
    payload = build_payload(
        body,
        dispatch_model=dispatch_model,
        forward_headers=forward_headers,
        api_key=api_key,
        api_base=api_base,
    )
    log.info("dispatch", shape=adapter.log_shape, model=client_model, dispatch_model=dispatch_model)
    result = coerce_to_dict(await dispatch(**payload))
    adapter.set_model_in_response(result, client_model)
    log_usage_from_body(adapter.shape, result, endpoint=adapter.endpoint, on_complete=on_complete)
    return result


async def proxy_translate(
    adapter: TranslateAdapter,
    request: dict[str, Any],
    *,
    dispatch_model: str,
    provider: str | None = None,
    completion: CompletionFn | None = None,
    forward_headers: dict[str, str] | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    on_complete: Callable[[Usage], None] | None = None,
) -> dict[str, Any]:
    """Generic non-streaming translate runner."""
    dispatch: Callable[..., Awaitable[Any]] = completion or adapter.default_dispatch
    return await _proxy_translate_inner(
        adapter,
        request,
        dispatch=dispatch,
        dispatch_model=dispatch_model,
        provider=provider,
        forward_headers=forward_headers,
        api_key=api_key,
        api_base=api_base,
        on_complete=on_complete,
    )


def stream_translate(
    adapter: TranslateAdapter,
    request: dict[str, Any],
    *,
    dispatch_model: str,
    provider: str | None = None,
    completion: CompletionFn | None = None,
    forward_headers: dict[str, str] | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    on_complete: Callable[[Usage], None] | None = None,
) -> AsyncIterator[bytes]:
    """Generic streaming translate runner.

    Sync-returning so a malformed request surfaces as 400 before any
    bytes are emitted (LiteLLM raises ``pydantic.ValidationError``).
    """
    dispatch: Callable[..., Awaitable[Any]] = completion or adapter.default_dispatch
    client_model = resolve_client_model(request.get("model", ""), provider, dispatch_model)
    body = request
    if adapter.preprocess_body is not None:
        body = adapter.preprocess_body(body, dispatch_model, client_model)
    payload = build_payload(
        {**body, "stream": True},
        dispatch_model=dispatch_model,
        forward_headers=forward_headers,
        api_key=api_key,
        api_base=api_base,
    )
    log.info(
        "dispatch",
        shape=adapter.log_shape,
        model=client_model,
        dispatch_model=dispatch_model,
        stream=True,
    )
    mutator = adapter.set_model_in_stream_event(payload, client_model)
    return tap_stream(
        rewrite_data_in_stream(adapter.stream_bytes_iter(payload, dispatch), mutator),
        adapter.shape,
        endpoint=adapter.endpoint,
        fallback_model=client_model,
        on_complete=on_complete,
    )
