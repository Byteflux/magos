"""``ShapeSpec`` dataclass + ``Shape`` literal — shared by every shape module."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

Shape = Literal["anthropic", "openai-chat", "openai-responses"]
CompressionProvider = Literal["anthropic", "openai"]


@dataclass(frozen=True, slots=True)
class ShapeSpec:
    """Flat, data-only description of one wire shape.

    Holds field locations (where messages / system / instructions live in
    the request body), endpoint paths, the Headroom compression-provider
    axis, and the response usage-key map. No methods, no behavior — the
    package's whole value is that consumers can read these as plain data
    instead of branching on shape names. See :mod:`magos.shapes`.
    """

    name: Shape
    endpoints: tuple[str, ...]
    compression_provider: CompressionProvider

    # Request body field locations. ``None`` when the shape does not have
    # a top-level field of that kind (OpenAI Chat encodes "system" inside
    # ``messages``; Anthropic + Chat have no ``instructions`` field).
    system_field: str | None
    messages_field: str | None
    instructions_field: str | None

    # Non-streaming response usage extraction. Each value is the path
    # (relative to the response body) to the integer token count.
    # ``cache_write`` is Anthropic-only; OpenAI shapes omit the key.
    usage_keys: Mapping[str, tuple[str, ...]]
