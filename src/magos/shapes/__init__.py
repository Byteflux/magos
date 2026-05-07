"""Wire-shape data: per-shape field locations and usage maps.

Three peer-level shape *values* live here (``ANTHROPIC``,
``OPENAI_CHAT``, ``OPENAI_RESPONSES``), each a frozen ``Shape`` instance
describing one wire format the proxy speaks. Plural package name
(rather than singular ``shape/``) reflects that these are
value-discriminated peer entities, not subclasses of a single
abstraction — the same pattern as ``sqlalchemy.dialects`` /
``pydantic.types`` / ``concurrent.futures``.

Consumers receive ``Shape`` instances from ``shape_for_endpoint`` (or
from configuration via ``shape_by_name``) and call methods on them
directly (``shape.extract_usage(body)``). String names live only on
``Shape.name`` and at the yaml boundary.
"""

from __future__ import annotations

from .anthropic import SPEC as ANTHROPIC
from .base import CompressionProvider, Shape, StreamEvent
from .openai_chat import SPEC as OPENAI_CHAT
from .openai_responses import SPEC as OPENAI_RESPONSES
from .usage import Usage

SHAPES: tuple[Shape, ...] = (ANTHROPIC, OPENAI_CHAT, OPENAI_RESPONSES)

_BY_ENDPOINT: dict[str, Shape] = {
    endpoint: shape for shape in SHAPES for endpoint in shape.endpoints
}
_BY_NAME: dict[str, Shape] = {shape.name: shape for shape in SHAPES}


def shape_for_endpoint(endpoint: str) -> Shape | None:
    """Map a routed endpoint to the response shape, or ``None`` for n/a."""
    return _BY_ENDPOINT.get(endpoint)


def shape_by_name(name: str) -> Shape | None:
    """Look up a ``Shape`` by its string name. Used at the yaml boundary."""
    return _BY_NAME.get(name)


__all__ = [
    "ANTHROPIC",
    "OPENAI_CHAT",
    "OPENAI_RESPONSES",
    "SHAPES",
    "CompressionProvider",
    "Shape",
    "StreamEvent",
    "Usage",
    "shape_by_name",
    "shape_for_endpoint",
]
