"""OpenAI -> Anthropic translation (reverse direction).

Covers the request shape used when a client speaks OpenAI but the upstream is
addressed in Anthropic terms, and the response shape used to return an
Anthropic upstream reply to an OpenAI client.
"""

from __future__ import annotations

import json
import time
from typing import Any

from magos.obs import traced
from magos.translation._models import (
    AnthropicResponse,
    AnthropicTextBlock,
    OpenAIContentPart,
    OpenAIImagePart,
    OpenAIMessage,
    OpenAIRequest,
    OpenAITextPart,
    OpenAIToolChoice,
    OpenAIToolChoiceFunction,
)
from magos.translation._shared import STOP_TO_FINISH, url_to_image_source


def _openai_content_to_str(content: str | list[OpenAIContentPart] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(p.text for p in content if isinstance(p, OpenAITextPart))


def _openai_tool_choice_to_anthropic(choice: OpenAIToolChoice) -> Any:
    if isinstance(choice, str):
        if choice == "auto":
            return {"type": "auto"}
        if choice == "required":
            return {"type": "any"}
        # "none" or anything else: drop. Caller decides whether to omit tools.
        return None
    return {"type": "tool", "name": choice.function.name}


def _openai_message_to_anthropic_blocks(msg: OpenAIMessage) -> list[dict[str, Any]]:
    """Convert a non-tool/non-system OpenAI message body into Anthropic blocks."""
    blocks: list[dict[str, Any]] = []

    if isinstance(msg.content, str):
        if msg.content:
            blocks.append({"type": "text", "text": msg.content})
    elif isinstance(msg.content, list):
        for part in msg.content:
            if isinstance(part, OpenAITextPart):
                blocks.append({"type": "text", "text": part.text})
            elif isinstance(part, OpenAIImagePart):
                blocks.append(
                    {
                        "type": "image",
                        "source": url_to_image_source(part.image_url.url),
                    }
                )

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

    return blocks


@traced("translation.request_openai_to_anthropic")
def request_openai_to_anthropic(payload: dict[str, Any]) -> dict[str, Any]:  # noqa: PLR0912
    """Translate an OpenAI Chat Completions request into an Anthropic Messages request."""
    req = OpenAIRequest.model_validate(payload)

    system_chunks: list[str] = []
    messages: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] = []

    def flush_tool_results() -> None:
        nonlocal pending_tool_results
        if pending_tool_results:
            messages.append({"role": "user", "content": pending_tool_results})
            pending_tool_results = []

    for omsg in req.messages:
        if omsg.role == "system":
            system_chunks.append(_openai_content_to_str(omsg.content))
            continue
        if omsg.role == "tool":
            pending_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": omsg.tool_call_id or "",
                    "content": _openai_content_to_str(omsg.content),
                }
            )
            continue
        flush_tool_results()
        blocks = _openai_message_to_anthropic_blocks(omsg)
        if not blocks:
            continue
        content: str | list[dict[str, Any]] = (
            blocks[0]["text"] if len(blocks) == 1 and blocks[0]["type"] == "text" else blocks
        )
        messages.append({"role": omsg.role, "content": content})
    flush_tool_results()

    out: dict[str, Any] = {
        "model": req.model,
        "max_tokens": req.max_tokens or req.max_completion_tokens or 1024,
        "messages": messages,
    }

    if system_chunks:
        out["system"] = "\n".join(c for c in system_chunks if c)

    if req.tools is not None:
        anth_tools: list[dict[str, Any]] = []
        for t in req.tools:
            tool_dict: dict[str, Any] = {
                "name": t.function.name,
                "input_schema": t.function.parameters,
            }
            if t.function.description is not None:
                tool_dict["description"] = t.function.description
            anth_tools.append(tool_dict)
        out["tools"] = anth_tools

    if req.tool_choice is not None:
        translated = _openai_tool_choice_to_anthropic(req.tool_choice)
        if translated is not None:
            out["tool_choice"] = translated

    if req.stop is not None:
        out["stop_sequences"] = [req.stop] if isinstance(req.stop, str) else req.stop
    if req.temperature is not None:
        out["temperature"] = req.temperature
    if req.top_p is not None:
        out["top_p"] = req.top_p
    if req.stream is not None:
        out["stream"] = req.stream
    if req.user is not None:
        out["metadata"] = {"user_id": req.user}

    return out


@traced("translation.response_anthropic_to_openai")
def response_anthropic_to_openai(
    payload: dict[str, Any],
    *,
    response_id: str | None = None,
    created: int | None = None,
) -> dict[str, Any]:
    """Translate an Anthropic Messages response into an OpenAI Chat Completions response.

    ``response_id`` and ``created`` are injectable for deterministic tests.
    Defaults: ``response_id`` derives from the Anthropic id; ``created`` uses
    the current unix epoch.
    """
    resp = AnthropicResponse.model_validate(payload)

    text_chunks: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in resp.content:
        if isinstance(block, AnthropicTextBlock):
            text_chunks.append(block.text)
        else:
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

    msg: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text_chunks) if text_chunks else None,
    }
    if tool_calls:
        msg["tool_calls"] = tool_calls

    return {
        "id": response_id or ("chatcmpl-" + resp.id.removeprefix("msg_")),
        "object": "chat.completion",
        "created": created if created is not None else int(time.time()),
        "model": resp.model,
        "choices": [
            {
                "index": 0,
                "message": msg,
                "finish_reason": STOP_TO_FINISH[resp.stop_reason],
            }
        ],
        "usage": {
            "prompt_tokens": resp.usage.input_tokens,
            "completion_tokens": resp.usage.output_tokens,
            "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens,
        },
    }


# Re-export for compatibility (used to live alongside in translation.py); kept
# here in the reverse module since OpenAI tool_choice handling is reverse-side.
__all__ = [
    "OpenAIToolChoiceFunction",
    "request_openai_to_anthropic",
    "response_anthropic_to_openai",
]
