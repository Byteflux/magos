"""Request abstraction shared by routing and dispatch.

Decouples the routing pipeline from FastAPI's ``Request`` so unit tests can
construct one directly without spinning up the app. ``raw_body`` is preserved
alongside the parsed ``body`` because byte-exact passthrough must forward the
original bytes when no rewrite touched the body.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

Endpoint = Literal[
    "/v1/messages",
    "/v1/messages/count_tokens",
    "/v1/chat/completions",
    "/v1/responses",
]

ENDPOINTS: tuple[Endpoint, ...] = (
    "/v1/messages",
    "/v1/messages/count_tokens",
    "/v1/chat/completions",
    "/v1/responses",
)


@dataclass(frozen=True, slots=True)
class RoutedRequest:
    """Inputs to the routing pipeline.

    ``headers`` keys are lowercased so matchers and rewrites can do
    case-insensitive lookups without re-normalising on every access.
    ``body_dirty`` is set by the rewrite stage when a body-touching op runs;
    passthrough dispatch consults it to decide between forwarding ``raw_body``
    verbatim (cache-preserving) and re-serialising ``body`` (cache-breaking).
    """

    endpoint: Endpoint
    headers: Mapping[str, str]
    body: Mapping[str, Any]
    raw_body: bytes
    body_dirty: bool = False
