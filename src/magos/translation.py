"""Anthropic <-> OpenAI shape translation.

Covers the static (non-streaming) request and response surfaces of both APIs:
text, images, tool definitions, tool_use / tool_calls, tool_result / tool role
messages, system prompts, stop sequences, and common sampling params.

Public functions:

- request_anthropic_to_openai   Anthropic Messages request  -> OpenAI Chat req
- response_openai_to_anthropic  OpenAI Chat response        -> Anthropic Messages resp
- request_openai_to_anthropic   OpenAI Chat request         -> Anthropic Messages request
- response_anthropic_to_openai  Anthropic Messages response -> OpenAI Chat response

Pydantic models validate at the boundary; output is plain dicts. Models use
``extra="ignore"`` to keep the goldens deterministic. Unknown client fields are
silently dropped, which is acceptable for the proxy until a passthrough policy
is added.
"""

from __future__ import annotations

import json
import secrets
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from magos.obs import traced

# ---------------------------------------------------------------------------
# Common literals
# ---------------------------------------------------------------------------

AnthropicRole = Literal["user", "assistant"]
OpenAIRole = Literal["system", "user", "assistant", "tool"]
OpenAIFinishReason = Literal["stop", "length", "tool_calls", "content_filter"]
AnthropicStopReason = Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


# ---------------------------------------------------------------------------
# Anthropic content blocks
# ---------------------------------------------------------------------------


class AnthropicTextBlock(_Frozen):
    type: Literal["text"]
    text: str


class AnthropicImageSourceBase64(_Frozen):
    type: Literal["base64"]
    media_type: str
    data: str


class AnthropicImageSourceUrl(_Frozen):
    type: Literal["url"]
    url: str


AnthropicImageSource = AnthropicImageSourceBase64 | AnthropicImageSourceUrl


class AnthropicImageBlock(_Frozen):
    type: Literal["image"]
    source: AnthropicImageSource


class AnthropicToolUseBlock(_Frozen):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]


class AnthropicToolResultBlock(_Frozen):
    type: Literal["tool_result"]
    tool_use_id: str
    content: str | list[AnthropicTextBlock]
    is_error: bool | None = None


AnthropicContentBlock = (
    AnthropicTextBlock | AnthropicImageBlock | AnthropicToolUseBlock | AnthropicToolResultBlock
)


class AnthropicMessage(_Frozen):
    role: AnthropicRole
    content: str | list[AnthropicContentBlock]


# ---------------------------------------------------------------------------
# Anthropic request models
# ---------------------------------------------------------------------------


class AnthropicTool(_Frozen):
    name: str
    description: str | None = None
    input_schema: dict[str, Any]


class AnthropicToolChoiceAuto(_Frozen):
    type: Literal["auto"]


class AnthropicToolChoiceAny(_Frozen):
    type: Literal["any"]


class AnthropicToolChoiceTool(_Frozen):
    type: Literal["tool"]
    name: str


AnthropicToolChoice = AnthropicToolChoiceAuto | AnthropicToolChoiceAny | AnthropicToolChoiceTool


class AnthropicMetadata(_Frozen):
    user_id: str | None = None


class AnthropicRequest(_Frozen):
    model: str
    max_tokens: int
    messages: list[AnthropicMessage]
    system: str | list[AnthropicTextBlock] | None = None
    tools: list[AnthropicTool] | None = None
    tool_choice: AnthropicToolChoice | None = None
    stop_sequences: list[str] | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stream: bool | None = None
    metadata: AnthropicMetadata | None = None


# ---------------------------------------------------------------------------
# Anthropic response models
# ---------------------------------------------------------------------------


class AnthropicUsage(_Frozen):
    input_tokens: int
    output_tokens: int


class AnthropicResponse(_Frozen):
    id: str
    type: Literal["message"]
    role: Literal["assistant"]
    model: str
    content: list[AnthropicTextBlock | AnthropicToolUseBlock]
    stop_reason: AnthropicStopReason
    stop_sequence: str | None = None
    usage: AnthropicUsage


# ---------------------------------------------------------------------------
# OpenAI content parts (for vision)
# ---------------------------------------------------------------------------


class OpenAITextPart(_Frozen):
    type: Literal["text"]
    text: str


class OpenAIImageUrl(_Frozen):
    url: str
    detail: Literal["auto", "low", "high"] | None = None


class OpenAIImagePart(_Frozen):
    type: Literal["image_url"]
    image_url: OpenAIImageUrl


OpenAIContentPart = OpenAITextPart | OpenAIImagePart


# ---------------------------------------------------------------------------
# OpenAI tool models
# ---------------------------------------------------------------------------


class OpenAIToolCallFunction(_Frozen):
    name: str
    arguments: str  # JSON-encoded


class OpenAIToolCall(_Frozen):
    id: str
    type: Literal["function"]
    function: OpenAIToolCallFunction


class OpenAIFunctionDef(_Frozen):
    name: str
    description: str | None = None
    parameters: dict[str, Any]


class OpenAITool(_Frozen):
    type: Literal["function"]
    function: OpenAIFunctionDef


class OpenAIToolChoiceFunctionInner(_Frozen):
    name: str


class OpenAIToolChoiceFunction(_Frozen):
    type: Literal["function"]
    function: OpenAIToolChoiceFunctionInner


# OpenAI tool_choice can be a string or an object
OpenAIToolChoice = str | OpenAIToolChoiceFunction


# ---------------------------------------------------------------------------
# OpenAI request models
# ---------------------------------------------------------------------------


class OpenAIMessage(_Frozen):
    role: OpenAIRole
    content: str | list[OpenAIContentPart] | None = None
    name: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None
    tool_call_id: str | None = None


class OpenAIRequest(_Frozen):
    model: str
    messages: list[OpenAIMessage]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    tools: list[OpenAITool] | None = None
    tool_choice: OpenAIToolChoice | None = None
    stop: str | list[str] | None = None
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    stream: bool | None = None
    user: str | None = None


# ---------------------------------------------------------------------------
# OpenAI response models
# ---------------------------------------------------------------------------


class OpenAIUsage(_Frozen):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIResponseMessage(_Frozen):
    role: Literal["assistant"]
    content: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None


class OpenAIChoice(_Frozen):
    index: int
    message: OpenAIResponseMessage
    finish_reason: OpenAIFinishReason


class OpenAIResponse(_Frozen):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: list[OpenAIChoice]
    usage: OpenAIUsage


# ---------------------------------------------------------------------------
# Reason mapping tables
# ---------------------------------------------------------------------------


_FINISH_TO_STOP: dict[OpenAIFinishReason, AnthropicStopReason] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
}

_STOP_TO_FINISH: dict[AnthropicStopReason, OpenAIFinishReason] = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
}


# ---------------------------------------------------------------------------
# Helpers: shared
# ---------------------------------------------------------------------------


def _system_to_str(system: str | list[AnthropicTextBlock] | None) -> str | None:
    if system is None:
        return None
    if isinstance(system, str):
        return system
    return "".join(b.text for b in system)


def _image_source_to_url(source: AnthropicImageSource) -> str:
    if isinstance(source, AnthropicImageSourceUrl):
        return source.url
    return f"data:{source.media_type};base64,{source.data}"


def _url_to_image_source(url: str) -> dict[str, Any]:
    if url.startswith("data:"):
        header, _, data = url[5:].partition(",")
        media_type = header.split(";", 1)[0]
        return {"type": "base64", "media_type": media_type, "data": data}
    return {"type": "url", "url": url}


# ---------------------------------------------------------------------------
# Anthropic -> OpenAI: request
# ---------------------------------------------------------------------------


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
                    "image_url": {"url": _image_source_to_url(block.source)},
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
    sys_str = _system_to_str(req.system)
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


# ---------------------------------------------------------------------------
# OpenAI -> Anthropic: response
# ---------------------------------------------------------------------------


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
        "stop_reason": _FINISH_TO_STOP[choice.finish_reason],
        "stop_sequence": None,
        "usage": {
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        },
    }


# ---------------------------------------------------------------------------
# OpenAI -> Anthropic: request (reverse)
# ---------------------------------------------------------------------------


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


def _openai_message_to_anthropic_blocks(
    msg: OpenAIMessage,
) -> list[dict[str, Any]]:
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
                        "source": _url_to_image_source(part.image_url.url),
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


# ---------------------------------------------------------------------------
# Anthropic -> OpenAI: response (reverse)
# ---------------------------------------------------------------------------


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
                "finish_reason": _STOP_TO_FINISH[resp.stop_reason],
            }
        ],
        "usage": {
            "prompt_tokens": resp.usage.input_tokens,
            "completion_tokens": resp.usage.output_tokens,
            "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens,
        },
    }


# ---------------------------------------------------------------------------
# OpenAI -> Anthropic: streaming
# ---------------------------------------------------------------------------


class AnthropicStreamTranslator:
    """Convert a stream of OpenAI Chat Completions chunks into Anthropic events.

    Anthropic's Messages streaming format is a stateful sequence:
    ``message_start``, then per content block ``content_block_start`` /
    ``content_block_delta`` / ``content_block_stop``, then ``message_delta``
    and ``message_stop``. This class buffers the cross-chunk state needed to
    emit that sequence from OpenAI's flatter chunk stream.

    Tool-call argument JSON is forwarded as ``input_json_delta`` slices; the
    client is responsible for reassembling and parsing.

    ``input_tokens`` is set to 0 in ``message_start`` since OpenAI streaming
    chunks do not carry prompt token counts; ``output_tokens`` is updated from
    a final usage chunk if present (e.g. ``stream_options.include_usage``).
    """

    def __init__(self, *, message_id: str | None = None) -> None:
        self._message_id = message_id or "msg_" + secrets.token_hex(12)
        self._started = False
        self._stopped = False
        self._model = ""
        self._next_block_index = 0
        self._open_block_index: int | None = None
        self._open_block_kind: str | None = None  # "text" | "tool_use"
        self._tool_index_map: dict[int, int] = {}  # OpenAI tool index -> Anthropic block index
        self._stop_reason: AnthropicStopReason | None = None
        self._output_tokens = 0

    def feed(self, chunk: dict[str, Any]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []

        if not self._started:
            self._started = True
            self._model = str(chunk.get("model", ""))
            events.append(
                {
                    "type": "message_start",
                    "message": {
                        "id": self._message_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": self._model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                }
            )

        usage = chunk.get("usage")
        if isinstance(usage, dict):
            ct = usage.get("completion_tokens")
            if isinstance(ct, int):
                self._output_tokens = ct

        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            tool_calls = delta.get("tool_calls")
            finish = choice.get("finish_reason")

            if content:
                events.extend(self._handle_text_delta(content))
            if tool_calls:
                events.extend(self._handle_tool_call_deltas(tool_calls))
            if finish:
                self._stop_reason = _FINISH_TO_STOP.get(finish, "end_turn")

        return events

    def finish(self) -> list[dict[str, Any]]:
        if self._stopped:
            return []
        self._stopped = True
        events: list[dict[str, Any]] = self._close_open_block()
        events.append(
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": self._stop_reason or "end_turn",
                    "stop_sequence": None,
                },
                "usage": {"output_tokens": self._output_tokens},
            }
        )
        events.append({"type": "message_stop"})
        return events

    def _handle_text_delta(self, text: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if self._open_block_kind != "text":
            events.extend(self._close_open_block())
            idx = self._next_block_index
            self._next_block_index += 1
            self._open_block_index = idx
            self._open_block_kind = "text"
            events.append(
                {
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {"type": "text", "text": ""},
                }
            )
        events.append(
            {
                "type": "content_block_delta",
                "index": self._open_block_index,
                "delta": {"type": "text_delta", "text": text},
            }
        )
        return events

    def _handle_tool_call_deltas(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for tc in tool_calls:
            oai_idx = int(tc.get("index", 0))
            fn = tc.get("function") or {}

            if oai_idx not in self._tool_index_map:
                events.extend(self._close_open_block())
                idx = self._next_block_index
                self._next_block_index += 1
                self._tool_index_map[oai_idx] = idx
                self._open_block_index = idx
                self._open_block_kind = "tool_use"
                events.append(
                    {
                        "type": "content_block_start",
                        "index": idx,
                        "content_block": {
                            "type": "tool_use",
                            "id": tc.get("id") or "",
                            "name": fn.get("name") or "",
                            "input": {},
                        },
                    }
                )

            args = fn.get("arguments")
            if args:
                events.append(
                    {
                        "type": "content_block_delta",
                        "index": self._tool_index_map[oai_idx],
                        "delta": {"type": "input_json_delta", "partial_json": args},
                    }
                )
        return events

    def _close_open_block(self) -> list[dict[str, Any]]:
        if self._open_block_index is None:
            return []
        ev: list[dict[str, Any]] = [{"type": "content_block_stop", "index": self._open_block_index}]
        self._open_block_index = None
        self._open_block_kind = None
        return ev
