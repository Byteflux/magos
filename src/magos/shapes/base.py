"""``Shape`` class — wire-format declaration with extraction methods.

Each shape (Anthropic Messages, OpenAI Chat, OpenAI Responses) is
declared once as a frozen ``Shape`` instance in its sibling module
(``anthropic.py``, ``openai_chat.py``, ``openai_responses.py``). The
class holds field locations (where messages / system / instructions
live in the request body), endpoint paths, the Headroom
compression-provider axis, and the response usage-key map. Methods
extract data from request and response bodies using the declared
field paths.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from magos.shapes.usage import Usage

CompressionProvider = Literal["anthropic", "openai"]


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """Where one streaming SSE event carries usage data.

    A shape may declare multiple events (Anthropic splits input vs
    output across ``message_start`` / ``message_delta``); the generic
    accumulator walks every entry whose ``event_name`` matches.
    ``event_name=None`` means "match any chunk" (OpenAI Chat puts usage
    on the terminal chunk regardless of event name).
    """

    event_name: str | None
    usage_path: tuple[str, ...]
    """Path to the usage dict within the event data."""

    model_path: tuple[str, ...] | None
    """Path to the model string within the event data, or ``None`` to skip."""

    fields: Mapping[str, tuple[str, ...]]
    """Canonical name (``input`` / ``output`` / ``cache_read`` / ``cache_write``)
    -> path within the usage dict."""


def _safe_int(value: Any) -> int:
    return value if isinstance(value, int) and value >= 0 else 0


def _walk(body: Any, path: tuple[str, ...]) -> Any:
    """Walk a dotted path through nested dicts, returning ``None`` on any miss."""
    cur: Any = body
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


@dataclass(frozen=True, slots=True)
class Shape:
    """Frozen declaration of one wire shape with extraction methods.

    Holds field locations (where messages / system / instructions live
    in the request body), endpoint paths, the Headroom
    compression-provider axis, and the response usage-key map. Methods
    extract data from request and response bodies using the declared
    field paths.
    """

    name: str
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

    # Streaming usage extraction. Each entry says "when this SSE event
    # fires, the usage dict is at this path; pull these fields from it".
    stream_events: tuple[StreamEvent, ...]

    def extract_usage(self, body: Any) -> Usage:
        """Extract a ``Usage`` from a non-streaming response body.

        Walks ``self.usage_keys`` to find the path to each canonical
        field; missing / non-int / negative values default to 0. Non-dict
        bodies return an empty ``Usage``.
        """
        from magos.shapes.usage import Usage  # avoid module-load cycle  # noqa: PLC0415

        if not isinstance(body, dict):
            return Usage()
        keys = self.usage_keys
        return Usage(
            input=_safe_int(_walk(body, keys["input"])) if "input" in keys else 0,
            output=_safe_int(_walk(body, keys["output"])) if "output" in keys else 0,
            cache_read=_safe_int(_walk(body, keys["cache_read"])) if "cache_read" in keys else 0,
            cache_write=_safe_int(_walk(body, keys["cache_write"])) if "cache_write" in keys else 0,
        )

    def extract_system_bytes(self, body: Mapping[str, Any]) -> bytes:
        """Extract system-prompt bytes via this shape's body-field declaration.

        Reads from ``system_field`` if present (top-level string or list of
        text blocks); otherwise reads from the first ``role=system`` entry
        in ``messages_field``. Returns empty bytes when neither field is
        set on this shape (e.g., OpenAI Responses).
        """
        if self.system_field is not None:
            return _from_top_level_field(body, self.system_field)
        if self.messages_field is not None:
            return _from_messages_field(body, self.messages_field)
        return b""


def _from_top_level_field(body: Mapping[str, Any], field: str) -> bytes:
    """Read a top-level system field that may be a string or a list of text blocks."""
    value = body.get(field, "")
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, list):
        parts: list[str] = [
            block["text"]
            for block in value
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
        return "".join(parts).encode("utf-8")
    return b""


def _from_messages_field(body: Mapping[str, Any], field: str) -> bytes:
    """Read the first ``role=system`` entry from a messages-style list."""
    messages = body.get(field, [])
    if not isinstance(messages, list):
        return b""
    for msg in messages:
        if not (isinstance(msg, dict) and msg.get("role") == "system"):
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content.encode("utf-8")
        if isinstance(content, list):
            parts: list[str] = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and isinstance(b.get("text"), str)
            ]
            return "".join(parts).encode("utf-8")
        return b""
    return b""
