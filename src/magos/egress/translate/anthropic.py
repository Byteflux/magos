"""``/v1/messages`` translate path via ``litellm.anthropic_messages``.

LiteLLM's Anthropic-unified endpoint accepts Anthropic-shape input and
emits Anthropic-shape output regardless of upstream provider. For
Anthropic-bound traffic, the body passes through verbatim. For
non-Anthropic dispatch models, we pre-translate Anthropic-only fields
(``output_config`` → OpenAI ``reasoning_effort`` / ``response_format``)
and drop fields the LiteLLM translator doesn't recognize, otherwise
they leak into the destination SDK's ``**kwargs`` and produce
``unexpected keyword argument`` errors. ``litellm.drop_params=True``
only catches params LiteLLM *recognizes* but the destination doesn't
support; unknown fields slip past it.
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

# Canonical Anthropic Messages body fields that LiteLLM's
# ``anthropic_messages`` translator knows how to map to non-Anthropic
# providers. Anything outside this set falls into ``**kwargs`` and leaks
# straight into the destination SDK on cross-provider routes.
_ANTHROPIC_MESSAGES_CANONICAL_FIELDS: frozenset[str] = frozenset(
    {
        "model",
        "messages",
        "max_tokens",
        "system",
        "temperature",
        "top_p",
        "top_k",
        "stop_sequences",
        "stream",
        "metadata",
        "tools",
        "tool_choice",
        "thinking",
        # OpenAI-shape fields produced by ``_translate_output_config``. They
        # ride through LiteLLM's ``anthropic_messages`` ``**kwargs`` to the
        # destination translator, which knows them.
        "reasoning_effort",
        "response_format",
    }
)

# Anthropic's ``output_config.effort`` accepts levels OpenAI's
# ``reasoning_effort`` does not (``xhigh`` and ``max``); clamp the high end so
# the destination doesn't reject the value. ``minimal`` is OpenAI-only and not
# emitted by Anthropic, so no inbound mapping is needed.
_ANTHROPIC_EFFORT_TO_OPENAI: dict[str, str] = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}


def _translate_output_config(body: dict[str, Any]) -> dict[str, Any]:
    """Map Anthropic ``output_config`` onto OpenAI-shape equivalents.

    Carries two operationally important fields:

    - ``effort`` -> OpenAI ``reasoning_effort`` (``xhigh``/``max`` clamp to ``high``)
    - ``format`` (json_schema) -> OpenAI ``response_format``

    Caller-supplied ``reasoning_effort`` / ``response_format`` win over
    the derived values.
    """
    cfg = body.get("output_config")
    if not isinstance(cfg, dict):
        return body
    out = {k: v for k, v in body.items() if k != "output_config"}
    effort = cfg.get("effort")
    if isinstance(effort, str) and "reasoning_effort" not in out:
        mapped = _ANTHROPIC_EFFORT_TO_OPENAI.get(effort)
        if mapped is not None:
            out["reasoning_effort"] = mapped
    fmt = cfg.get("format")
    if isinstance(fmt, dict) and fmt.get("type") == "json_schema" and "response_format" not in out:
        # Anthropic nests the schema directly under ``format``; OpenAI wraps
        # it in a ``json_schema`` object. The schema body itself is identical.
        schema = fmt.get("schema")
        if isinstance(schema, dict):
            out["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": fmt.get("name", "response"),
                    "schema": schema,
                    **({"strict": True} if fmt.get("strict") else {}),
                },
            }
    return out


def _strip_anthropic_extras(body: dict[str, Any], dispatch_model: str) -> dict[str, Any]:
    """Drop Anthropic-only body fields when dispatching to a non-Anthropic upstream.

    ``output_config`` is translated to OpenAI equivalents first so
    structured-output / reasoning-effort behavior carries over;
    remaining Anthropic-only fields (``context_management``, future
    additions) are dropped. Anthropic-bound traffic is left untouched.
    """
    if dispatch_model.startswith("anthropic/"):
        return body
    body = _translate_output_config(body)
    extras = set(body) - _ANTHROPIC_MESSAGES_CANONICAL_FIELDS
    if not extras:
        return body
    log.info(
        "anthropic.dropped_unknown_fields",
        model=dispatch_model,
        fields=sorted(extras),
    )
    return {k: v for k, v in body.items() if k in _ANTHROPIC_MESSAGES_CANONICAL_FIELDS}


@traced("proxy.anthropic_messages")
async def proxy_anthropic_messages(
    anthropic_request: dict[str, Any],
    *,
    dispatch_model: str,
    completion: CompletionFn | None = None,
    forward_headers: dict[str, str] | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> dict[str, Any]:
    """Round-trip an Anthropic Messages request through ``litellm.anthropic_messages``."""
    dispatch: Callable[..., Awaitable[Any]] = completion or litellm.anthropic_messages
    payload = build_payload(
        _strip_anthropic_extras(anthropic_request, dispatch_model),
        dispatch_model=dispatch_model,
        forward_headers=forward_headers,
        api_key=api_key,
        api_base=api_base,
    )
    log.info("dispatch", shape="anthropic", model=dispatch_model)
    body = coerce_to_dict(await dispatch(**payload))
    log_usage_from_body("anthropic", body, endpoint="/v1/messages")
    return body


def stream_anthropic_messages(
    anthropic_request: dict[str, Any],
    *,
    dispatch_model: str,
    completion: CompletionFn | None = None,
    forward_headers: dict[str, str] | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> AsyncIterator[bytes]:
    """Stream an Anthropic Messages request via ``litellm.anthropic_messages``.

    Returned as a regular function (not an async generator) so request
    validation runs synchronously: a malformed request raises
    ``pydantic.ValidationError`` (inside LiteLLM) before any response bytes
    are emitted, and the server can surface it as 400 rather than mid-stream.

    LiteLLM yields raw Anthropic SSE bytes (``event: message_start``,
    ``content_block_delta``, etc.); we forward them verbatim.
    """
    dispatch: Callable[..., Awaitable[Any]] = completion or litellm.anthropic_messages
    payload = build_payload(
        {**_strip_anthropic_extras(anthropic_request, dispatch_model), "stream": True},
        dispatch_model=dispatch_model,
        forward_headers=forward_headers,
        api_key=api_key,
        api_base=api_base,
    )
    log.info("dispatch", shape="anthropic", model=dispatch_model, stream=True)
    return tap_stream(
        _anthropic_bytes_iter(payload, dispatch),
        "anthropic",
        endpoint="/v1/messages",
        fallback_model=dispatch_model,
    )


async def _anthropic_bytes_iter(
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
