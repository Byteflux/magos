"""Anthropic -> OpenAI translation (forward direction).

Covers the request shape used to dispatch Anthropic Messages calls through an
OpenAI-shape upstream, and the response shape used to return that upstream's
reply to the Anthropic client.
"""

from __future__ import annotations

import json
from typing import Any

from magos.obs import traced
from magos.translation._models import (
    AnthropicImageBlock,
    AnthropicMessage,
    AnthropicRequest,
    AnthropicTextBlock,
    AnthropicTool,
    AnthropicToolChoice,
    AnthropicToolChoiceAny,
    AnthropicToolChoiceAuto,
    AnthropicToolResultBlock,
    AnthropicToolUseBlock,
    OpenAIResponse,
)
from magos.translation._shared import (
    FINISH_TO_STOP,
    image_source_to_url,
    system_to_str,
)


def _anthropic_tool_to_openai(tool: AnthropicTool) -> dict[str, Any]:
    fn: dict[str, Any] = {"name": tool.name, "parameters": tool.input_schema}
    if tool.description is not None:
        fn["description"] = tool.description
    return {"type": "function", "function": fn}


def _anthropic_tool_choice_to_openai(choice: AnthropicToolChoice) -> Any:
    if isinstance(choice, AnthropicToolChoiceAuto):
        return "auto"
    if isinstance(choice, AnthropicToolChoiceAny):
        return "required"
    return {"type": "function", "function": {"name": choice.name}}


def _anthropic_message_to_openai(msg: AnthropicMessage) -> list[dict[str, Any]]:  # noqa: PLR0912
    """Translate one Anthropic message into 1+ OpenAI messages.

    Tool result blocks are split out as separate ``role: tool`` messages and
    emitted before the main user/assistant message that contained them.
    """
    if isinstance(msg.content, str):
        return [{"role": msg.role, "content": msg.content}]

    text_chunks: list[str] = []
    parts: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    tool_messages: list[dict[str, Any]] = []
    has_image = False

    for block in msg.content:
        if isinstance(block, AnthropicTextBlock):
            text_chunks.append(block.text)
            parts.append({"type": "text", "text": block.text})
        elif isinstance(block, AnthropicImageBlock):
            has_image = True
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_source_to_url(block.source)},
                }
            )
        elif isinstance(block, AnthropicToolUseBlock):
            tool_calls.append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                }
            )
        elif isinstance(block, AnthropicToolResultBlock):
            tr_content: str | list[dict[str, Any]]
            if isinstance(block.content, str):
                tr_content = block.content
            else:
                tr_content = [{"type": "text", "text": b.text} for b in block.content]
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": block.tool_use_id,
                    "content": tr_content,
                }
            )

    out: list[dict[str, Any]] = list(tool_messages)

    main_content: str | list[dict[str, Any]] | None
    if has_image:
        main_content = parts
    elif text_chunks:
        main_content = "".join(text_chunks)
    else:
        main_content = None

    if main_content is not None or tool_calls:
        msg_dict: dict[str, Any] = {"role": msg.role, "content": main_content}
        if tool_calls and msg.role == "assistant":
            msg_dict["tool_calls"] = tool_calls
        out.append(msg_dict)

    return out


@traced("translation.request_anthropic_to_openai")
def request_anthropic_to_openai(payload: dict[str, Any]) -> dict[str, Any]:
    """Translate an Anthropic Messages request into an OpenAI Chat Completions request."""
    req = AnthropicRequest.model_validate(payload)

    messages: list[dict[str, Any]] = []
    sys_str = system_to_str(req.system)
    if sys_str:
        messages.append({"role": "system", "content": sys_str})
    for msg in req.messages:
        messages.extend(_anthropic_message_to_openai(msg))

    out: dict[str, Any] = {
        "model": req.model,
        "max_tokens": req.max_tokens,
        "messages": messages,
    }
    if req.tools is not None:
        out["tools"] = [_anthropic_tool_to_openai(t) for t in req.tools]
    if req.tool_choice is not None:
        out["tool_choice"] = _anthropic_tool_choice_to_openai(req.tool_choice)
    if req.stop_sequences:
        out["stop"] = req.stop_sequences
    if req.temperature is not None:
        out["temperature"] = req.temperature
    if req.top_p is not None:
        out["top_p"] = req.top_p
    # top_k intentionally dropped: OpenAI has no analogue.
    if req.stream is not None:
        out["stream"] = req.stream
    if req.metadata is not None and req.metadata.user_id is not None:
        out["user"] = req.metadata.user_id
    return out


@traced("translation.response_openai_to_anthropic")
def response_openai_to_anthropic(payload: dict[str, Any]) -> dict[str, Any]:
    """Translate an OpenAI Chat Completions response into an Anthropic Messages response."""
    resp = OpenAIResponse.model_validate(payload)
    if not resp.choices:
        raise ValueError("OpenAI response has no choices")
    choice = resp.choices[0]
    msg = choice.message

    blocks: list[dict[str, Any]] = []
    if msg.content:
        blocks.append({"type": "text", "text": msg.content})
    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": args,
                }
            )

    return {
        "id": "msg_" + resp.id.removeprefix("chatcmpl-"),
        "type": "message",
        "role": "assistant",
        "model": resp.model,
        "content": blocks,
        "stop_reason": FINISH_TO_STOP[choice.finish_reason],
        "stop_sequence": None,
        "usage": {
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        },
    }
