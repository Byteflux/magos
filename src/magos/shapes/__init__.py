"""Wire-shape data: per-shape field locations and usage maps.

Three values: ``anthropic``, ``openai-chat``, ``openai-responses`` — the
distinct request/response/streaming wire formats the proxy speaks.

This package is **data only**. Each shape module exposes a frozen
``ShapeSpec`` describing where messages / system / instructions live in
the body, the response usage-key map, and the Headroom compression
provider. Consumers (usage extraction, session-id derivation, cache-mode
compression, etc.) read these as plain data rather than branching on
shape names.

The discipline that keeps this from rotting: no functions that take a
``Shape`` and *do work* belong here. Behaviour stays in its concern; only
flat lookups live here. See ``CLAUDE.md`` for the rationale.
"""

from __future__ import annotations

from ._spec import CompressionProvider, Shape, ShapeSpec
from .anthropic import SPEC as ANTHROPIC
from .openai_chat import SPEC as OPENAI_CHAT
from .openai_responses import SPEC as OPENAI_RESPONSES

SHAPES: dict[Shape, ShapeSpec] = {
    ANTHROPIC.name: ANTHROPIC,
    OPENAI_CHAT.name: OPENAI_CHAT,
    OPENAI_RESPONSES.name: OPENAI_RESPONSES,
}

_ENDPOINT_TO_SHAPE: dict[str, Shape] = {
    endpoint: spec.name for spec in SHAPES.values() for endpoint in spec.endpoints
}


def shape_for_endpoint(endpoint: str) -> Shape | None:
    """Map a routed endpoint to the response shape, or ``None`` for n/a."""
    return _ENDPOINT_TO_SHAPE.get(endpoint)


__all__ = [
    "ANTHROPIC",
    "OPENAI_CHAT",
    "OPENAI_RESPONSES",
    "SHAPES",
    "CompressionProvider",
    "Shape",
    "ShapeSpec",
    "shape_for_endpoint",
]
