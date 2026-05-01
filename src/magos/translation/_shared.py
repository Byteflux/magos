"""Helpers and reason mapping tables shared across forward, reverse, and streaming modules."""

from __future__ import annotations

from typing import Any

from magos.translation._models import (
    AnthropicImageSource,
    AnthropicImageSourceUrl,
    AnthropicStopReason,
    AnthropicTextBlock,
    OpenAIFinishReason,
)

# ---------------------------------------------------------------------------
# Reason mapping tables
# ---------------------------------------------------------------------------

FINISH_TO_STOP: dict[OpenAIFinishReason, AnthropicStopReason] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
}

STOP_TO_FINISH: dict[AnthropicStopReason, OpenAIFinishReason] = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def system_to_str(system: str | list[AnthropicTextBlock] | None) -> str | None:
    if system is None:
        return None
    if isinstance(system, str):
        return system
    return "".join(b.text for b in system)


def image_source_to_url(source: AnthropicImageSource) -> str:
    if isinstance(source, AnthropicImageSourceUrl):
        return source.url
    return f"data:{source.media_type};base64,{source.data}"


def url_to_image_source(url: str) -> dict[str, Any]:
    if url.startswith("data:"):
        header, _, data = url[5:].partition(",")
        media_type = header.split(";", 1)[0]
        return {"type": "base64", "media_type": media_type, "data": data}
    return {"type": "url", "url": url}
