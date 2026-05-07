"""Choose Anthropic-native vs OpenAI-translated dispatch.

Anthropic upstream uses `litellm.anthropic_messages` directly;
everything else goes via `acompletion` + Anthropic<->OpenAI
translation because `anthropic_messages` leaks the LiteLLM provider
prefix into the outbound model id and gets rejected by non-Anthropic
upstreams.
"""

from __future__ import annotations

from typing import Any, cast

import litellm
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    AnthropicAdapter,
    LiteLLMAnthropicMessagesAdapter,
)
from litellm.types.utils import ModelResponse

_OPENAI_EXTRA_FIELDS = ("reasoning_effort", "response_format")


async def _dispatch_anthropic_messages(**payload: Any) -> Any:
    model = payload.get("model", "")
    try:
        _, provider, _, _ = litellm.get_llm_provider(model=model)
    except Exception:
        provider = None
    if provider == "anthropic":
        return await litellm.anthropic_messages(**payload)
    return await _via_acompletion(payload)


async def _via_acompletion(payload: dict[str, Any]) -> Any:
    """Anthropic->OpenAI translation + `litellm.acompletion`; preserves
    the OpenAI extras (`reasoning_effort`, `response_format`) the
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
