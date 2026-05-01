"""Anthropic <-> OpenAI shape translation.

Pure functions over JSON-shaped dicts. Pydantic models validate the inbound
payload at the boundary; output is plain dicts so callers can serialize without
re-running model machinery.

Scope is intentionally narrow: only what the current golden fixtures exercise.
Extend the models (and the goldens) together as new cases land.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from magos.obs import traced

AnthropicRole = Literal["user", "assistant"]
OpenAIRole = Literal["system", "user", "assistant"]
OpenAIFinishReason = Literal["stop", "length", "tool_calls", "content_filter"]
AnthropicStopReason = Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class AnthropicTextBlock(_Model):
    type: Literal["text"]
    text: str


class AnthropicInputMessage(_Model):
    role: AnthropicRole
    content: str | list[AnthropicTextBlock]


class AnthropicRequest(_Model):
    model: str
    max_tokens: int
    messages: list[AnthropicInputMessage]
    system: str | None = None


class OpenAIResponseMessage(_Model):
    role: Literal["assistant"]
    content: str


class OpenAIChoice(_Model):
    index: int
    message: OpenAIResponseMessage
    finish_reason: OpenAIFinishReason


class OpenAIUsage(_Model):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIResponse(_Model):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: list[OpenAIChoice]
    usage: OpenAIUsage


_FINISH_TO_STOP: dict[OpenAIFinishReason, AnthropicStopReason] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
}


def _flatten_content(content: str | list[AnthropicTextBlock]) -> str:
    if isinstance(content, str):
        return content
    return "".join(block.text for block in content)


@traced("translation.request_anthropic_to_openai")
def request_anthropic_to_openai(payload: dict[str, Any]) -> dict[str, Any]:
    """Translate an Anthropic Messages request into an OpenAI Chat Completions request."""
    req = AnthropicRequest.model_validate(payload)
    messages: list[dict[str, Any]] = []
    if req.system is not None:
        messages.append({"role": "system", "content": req.system})
    for msg in req.messages:
        messages.append({"role": msg.role, "content": _flatten_content(msg.content)})
    return {
        "model": req.model,
        "max_tokens": req.max_tokens,
        "messages": messages,
    }


@traced("translation.response_openai_to_anthropic")
def response_openai_to_anthropic(payload: dict[str, Any]) -> dict[str, Any]:
    """Translate an OpenAI Chat Completions response into an Anthropic Messages response."""
    resp = OpenAIResponse.model_validate(payload)
    if not resp.choices:
        raise ValueError("OpenAI response has no choices")
    choice = resp.choices[0]
    msg_id = "msg_" + resp.id.removeprefix("chatcmpl-")
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": resp.model,
        "content": [{"type": "text", "text": choice.message.content}],
        "stop_reason": _FINISH_TO_STOP[choice.finish_reason],
        "stop_sequence": None,
        "usage": {
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        },
    }
