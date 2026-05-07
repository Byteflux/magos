"""``/v1/messages`` translate path via ``litellm.anthropic_messages``.

Anthropic-shape in, Anthropic-shape out across upstreams. Non-Anthropic
dispatch pre-translates ``output_config`` to OpenAI extras and drops
unknown Anthropic-only fields (``litellm.drop_params`` doesn't catch
fields LiteLLM doesn't recognize). See ``docs/architecture/translation.md``.
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

from magos.egress.translate.runner import TranslateAdapter
from magos.egress.translate.sse import sse_named_event
from magos.telemetry import get_logger

log = get_logger("magos.egress.translate")

# Fields LiteLLM's ``anthropic_messages`` translator maps to non-Anthropic
# providers; anything else leaks via ``**kwargs`` into the destination SDK.
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
        # OpenAI-shape extras produced by ``_translate_output_config``;
        # ride ``**kwargs`` to the destination translator.
        "reasoning_effort",
        "response_format",
    }
)

# Anthropic accepts ``xhigh``/``max``; OpenAI's ``reasoning_effort`` tops
# out at ``high``. ``minimal`` is OpenAI-only and never inbound.
_ANTHROPIC_EFFORT_TO_OPENAI: dict[str, str] = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}


def _translate_output_config(body: dict[str, Any]) -> dict[str, Any]:
    """Map Anthropic ``output_config`` to OpenAI ``reasoning_effort`` / ``response_format``.

    Caller-supplied ``reasoning_effort`` / ``response_format`` win over
    the derived values. ``xhigh``/``max`` effort clamps to ``high``.
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


# Top-level fields that legitimately carry JSON Schema; ``messages``
# never does, so excluding it avoids scanning the bulk of every body.
_SCHEMA_BEARING_FIELDS: tuple[str, ...] = ("tools", "tool_choice", "response_format")


def _coerce_empty_additional_properties(body: dict[str, Any]) -> dict[str, Any]:
    """Replace ``additionalProperties: {}`` with ``true`` in schema-bearing fields.

    Semantically identical per JSON Schema, but some openai-compatible
    upstreams (Vultr) reject the empty-object form. Walks only schema-
    bearing top-level fields and shares storage on unchanged subtrees.
    """
    updates: dict[str, Any] = {}
    for field in _SCHEMA_BEARING_FIELDS:
        if field not in body:
            continue
        new_value = _coerce_empty_ap(body[field])
        if new_value is not body[field]:
            updates[field] = new_value
    if not updates:
        return body
    log.info("anthropic.coerced_empty_additional_properties")
    return {**body, **updates}


def _coerce_empty_ap(value: Any) -> Any:
    """Return ``value`` with empty-dict ``additionalProperties`` coerced to True.

    Returns the input by reference if no coercion was needed -- the caller
    uses ``is`` to detect changes, so unchanged subtrees share storage.
    """
    if isinstance(value, dict):
        new_pairs: dict[str, Any] | None = None
        for key, child in value.items():
            if key == "additionalProperties" and isinstance(child, dict) and not child:
                new_pairs = new_pairs or dict(value)
                new_pairs[key] = True
                continue
            new_child = _coerce_empty_ap(child)
            if new_child is not child:
                new_pairs = new_pairs or dict(value)
                new_pairs[key] = new_child
        return new_pairs if new_pairs is not None else value
    if isinstance(value, list):
        new_items: list[Any] | None = None
        for index, item in enumerate(value):
            new_item = _coerce_empty_ap(item)
            if new_item is not item:
                new_items = new_items or list(value)
                new_items[index] = new_item
        return new_items if new_items is not None else value
    return value


def _strip_anthropic_extras(
    body: dict[str, Any], dispatch_model: str, *, client_model: str
) -> dict[str, Any]:
    """Translate ``output_config``, coerce empty ``additionalProperties``,
    drop unknown Anthropic-only fields. No-op for Anthropic-bound traffic.
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
        model=client_model,
        dispatch_model=dispatch_model,
        fields=sorted(extras),
    )
    return {k: v for k, v in body.items() if k in _ANTHROPIC_MESSAGES_CANONICAL_FIELDS}


async def _dispatch_anthropic_messages(**payload: Any) -> Any:
    """Anthropic upstream uses ``litellm.anthropic_messages`` directly;
    everything else goes via ``acompletion`` + Anthropic<->OpenAI translation
    because ``anthropic_messages`` leaks the LiteLLM provider prefix into
    the outbound model id and gets rejected by non-Anthropic upstreams.
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
    """Anthropic->OpenAI translation + ``litellm.acompletion``; preserves
    the OpenAI extras (``reasoning_effort``, ``response_format``) the
    upstream adapter would otherwise drop.
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


def _anthropic_set_model_in_response(body: dict[str, Any], client_model: str) -> None:
    body["model"] = client_model


def _anthropic_set_model_in_stream_event(
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


def _anthropic_preprocess_body(
    body: dict[str, Any], dispatch_model: str, client_model: str
) -> dict[str, Any]:
    return _strip_anthropic_extras(body, dispatch_model, client_model=client_model)


ADAPTER = TranslateAdapter(
    shape="anthropic",
    endpoint="/v1/messages",
    default_dispatch=_dispatch_anthropic_messages,
    set_model_in_response=_anthropic_set_model_in_response,
    set_model_in_stream_event=_anthropic_set_model_in_stream_event,
    stream_bytes_iter=_anthropic_bytes_iter,
    log_shape="anthropic",
    preprocess_body=_anthropic_preprocess_body,
)
