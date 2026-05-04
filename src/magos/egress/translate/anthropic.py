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
from typing import Any, cast

import litellm
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    AnthropicAdapter,
    LiteLLMAnthropicMessagesAdapter,
)
from litellm.types.utils import ModelResponse

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


def _coerce_empty_additional_properties(body: dict[str, Any]) -> dict[str, Any]:
    """Replace ``additionalProperties: {}`` with ``additionalProperties: true``.

    JSON Schema treats both as semantically identical ("any extra property
    allowed, with no constraints"), but some openai-compatible upstreams
    (notably Vultr) misvalidate the empty-object form -- their metaschema
    validator reports it as ``[]`` and rejects the request with
    ``[] is not of type 'object', 'boolean'``. ``true`` flows through every
    validator we've tested and carries the same semantics.

    Walks the entire body so the coercion catches schemas wherever they
    appear -- ``tools[*].input_schema`` is the common case but
    ``response_format`` (post translation), ``json_schema`` blocks, and
    nested ``properties`` / ``items`` schemas can each carry the empty-dict
    form.
    """
    out, changed = _walk_coerce_empty_ap(body)
    if not changed or not isinstance(out, dict):
        return body
    log.info("anthropic.coerced_empty_additional_properties")
    return out


def _walk_coerce_empty_ap(value: Any) -> tuple[Any, bool]:
    if isinstance(value, dict):
        changed = False
        out: dict[str, Any] = {}
        for key, child in value.items():
            if key == "additionalProperties" and child == {}:
                out[key] = True
                changed = True
            else:
                new_child, child_changed = _walk_coerce_empty_ap(child)
                if child_changed:
                    changed = True
                out[key] = new_child
        return out, changed
    if isinstance(value, list):
        changed = False
        out_list: list[Any] = []
        for item in value:
            new_item, item_changed = _walk_coerce_empty_ap(item)
            if item_changed:
                changed = True
            out_list.append(new_item)
        return out_list, changed
    return value, False


def _strip_anthropic_extras(body: dict[str, Any], dispatch_model: str) -> dict[str, Any]:
    """Drop Anthropic-only body fields when dispatching to a non-Anthropic upstream.

    ``output_config`` is translated to OpenAI equivalents first so
    structured-output / reasoning-effort behavior carries over; tool
    ``input_schema`` blocks have ``additionalProperties: {}`` coerced to
    ``true`` (semantically identical, sidesteps a Vultr metaschema-validator
    bug); remaining Anthropic-only fields (``context_management``, future
    additions) are dropped. Anthropic-bound traffic is left untouched.
    """
    if dispatch_model.startswith("anthropic/"):
        return body
    body = _translate_output_config(body)
    body = _coerce_empty_additional_properties(body)
    extras = set(body) - _ANTHROPIC_MESSAGES_CANONICAL_FIELDS
    if not extras:
        return body
    log.info(
        "anthropic.dropped_unknown_fields",
        model=dispatch_model,
        fields=sorted(extras),
    )
    return {k: v for k, v in body.items() if k in _ANTHROPIC_MESSAGES_CANONICAL_FIELDS}


async def _dispatch_anthropic_messages(**payload: Any) -> Any:
    """Wrap ``litellm.anthropic_messages`` to work around an upstream bug.

    LiteLLM's ``anthropic_messages`` adapter chain leaks the LiteLLM
    provider prefix into the outbound request body when dispatching to
    non-Anthropic providers (OpenRouter and other custom-OpenAI-shape
    upstreams). E.g. ``model="openrouter/qwen/qwen3-coder"`` gets sent
    to OpenRouter as ``"openrouter/qwen/qwen3-coder"`` (with the
    prefix), which OpenRouter rejects with 400 *not a valid model ID*.
    ``litellm.acompletion`` strips the prefix correctly.

    Detect non-Anthropic dispatch and route through ``acompletion`` +
    manual Anthropic↔OpenAI body translation; keep Anthropic-bound
    traffic on the fast pass-through.
    """
    model = payload.get("model", "")
    try:
        _, provider, _, _ = litellm.get_llm_provider(model=model)
    except Exception:
        provider = None
    if provider == "anthropic":
        return await litellm.anthropic_messages(**payload)
    return await _via_acompletion(payload)


_OPENAI_EXTRA_FIELDS = ("reasoning_effort", "response_format")


async def _via_acompletion(payload: dict[str, Any]) -> Any:
    """Manual Anthropic→OpenAI translation + ``litellm.acompletion`` dispatch.

    Used when ``litellm.anthropic_messages`` would mishandle the model
    name (see ``_dispatch_anthropic_messages``). Preserves the OpenAI-
    shape extras magos injects via ``output_config`` translation
    (``reasoning_effort``, ``response_format``) which the upstream
    LiteLLM adapter would otherwise drop.
    """
    request_adapter = LiteLLMAnthropicMessagesAdapter()  # type: ignore[no-untyped-call]
    response_adapter = AnthropicAdapter()
    payload = dict(payload)
    api_base = payload.pop("api_base", None)
    api_key = payload.pop("api_key", None)
    extra_headers = payload.pop("extra_headers", None)
    stream = bool(payload.pop("stream", False))
    extras = {k: payload.pop(k) for k in _OPENAI_EXTRA_FIELDS if k in payload}

    openai_request, tool_name_mapping = request_adapter.translate_anthropic_to_openai(
        anthropic_message_request=payload  # type: ignore[arg-type]
    )
    completion_kwargs: dict[str, Any] = dict(openai_request)
    completion_kwargs.update(extras)
    if api_base is not None:
        completion_kwargs["api_base"] = api_base
    if api_key is not None:
        completion_kwargs["api_key"] = api_key
    if extra_headers is not None:
        completion_kwargs["extra_headers"] = extra_headers
    if stream:
        completion_kwargs["stream"] = True
        completion_kwargs["stream_options"] = {"include_usage": True}

    response = await litellm.acompletion(**completion_kwargs)
    if stream:
        return response_adapter.translate_completion_output_params_streaming(
            response,
            model=str(payload.get("model", "")),
            tool_name_mapping=tool_name_mapping,
        )
    return response_adapter.translate_completion_output_params(
        cast(ModelResponse, response),
        tool_name_mapping=tool_name_mapping,
    )


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
    dispatch: Callable[..., Awaitable[Any]] = completion or _dispatch_anthropic_messages
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
    """Stream an Anthropic Messages request via ``_dispatch_anthropic_messages``.

    Returned as a regular function (not an async generator) so request
    validation runs synchronously: a malformed request raises
    ``pydantic.ValidationError`` (inside LiteLLM) before any response bytes
    are emitted, and the server can surface it as 400 rather than mid-stream.

    LiteLLM yields raw Anthropic SSE bytes (``event: message_start``,
    ``content_block_delta``, etc.); we forward them verbatim.
    """
    dispatch: Callable[..., Awaitable[Any]] = completion or _dispatch_anthropic_messages
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
