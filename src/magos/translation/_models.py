"""Pydantic models for Anthropic Messages and OpenAI Chat Completions.

All models share a frozen base with ``extra="ignore"`` so unknown client
fields are silently dropped. Output of translation functions is plain dicts;
these models exist only to validate input at the boundary.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

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


class AnthropicCountTokensRequest(_Frozen):
    """Request body for ``POST /v1/messages/count_tokens``.

    Mirrors Anthropic's public surface: ``max_tokens`` is **not** required (the
    count_tokens endpoint estimates input tokens only and ignores generation
    limits). Sampling params and ``stream`` are also out of scope.
    """

    model: str
    messages: list[AnthropicMessage]
    system: str | list[AnthropicTextBlock] | None = None
    tools: list[AnthropicTool] | None = None
    tool_choice: AnthropicToolChoice | None = None


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
