"""Request abstraction shared by routing and dispatch. See ``docs/routing/pipeline.md``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

Endpoint = Literal[
    "/v1/messages",
    "/v1/messages/count_tokens",
    "/v1/chat/completions",
    "/v1/responses",
    "/v1/responses/{id}",
    "/v1/responses/{id}/input_items",
]

ENDPOINTS: tuple[Endpoint, ...] = (
    "/v1/messages",
    "/v1/messages/count_tokens",
    "/v1/chat/completions",
    "/v1/responses",
    "/v1/responses/{id}",
    "/v1/responses/{id}/input_items",
)

HttpMethod = Literal["GET", "POST", "DELETE"]


@dataclass(frozen=True, slots=True)
class RoutedRequest:
    """Inputs to the routing pipeline. See ``docs/routing/pipeline.md``.

    ``headers`` keys are lowercased. ``body_dirty`` flips on body-touching
    rewrites. ``actual_path`` overrides ``endpoint`` for upstream forwarding
    on templated paths.
    """

    endpoint: Endpoint
    headers: Mapping[str, str]
    body: Mapping[str, Any]
    raw_body: bytes
    body_dirty: bool = False
    method: HttpMethod = "POST"
    actual_path: str | None = None

    @property
    def forward_path(self) -> str:
        return self.actual_path if self.actual_path is not None else self.endpoint
