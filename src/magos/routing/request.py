"""Request abstraction shared by routing and dispatch.

Decouples the routing pipeline from FastAPI's ``Request`` so unit tests can
construct one directly without spinning up the app. ``raw_body`` is preserved
alongside the parsed ``body`` because byte-exact passthrough must forward the
original bytes when no rewrite touched the body.

For auxiliary OpenAI Responses endpoints (retrieve / cancel / list input
items), ``endpoint`` carries the *templated* path (e.g.
``/v1/responses/{id}``) so match expressions stay stable across response IDs,
while ``actual_path`` carries the concrete inbound path used by the
dispatcher when forwarding upstream. ``method`` distinguishes GET / DELETE
from the POST default; non-POST traffic must be routed via ``mode:
passthrough`` because litellm has no translate-mode equivalent.
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
    """Inputs to the routing pipeline.

    ``headers`` keys are lowercased so matchers and rewrites can do
    case-insensitive lookups without re-normalising on every access.
    ``body_dirty`` is set by the rewrite stage when a body-touching op runs;
    passthrough dispatch consults it to decide between forwarding ``raw_body``
    verbatim (cache-preserving) and re-serialising ``body`` (cache-breaking).
    ``actual_path`` overrides ``endpoint`` for upstream forwarding when the
    inbound path is templated (e.g. ``/v1/responses/{id}`` -> the concrete
    ``/v1/responses/resp_abc123``). ``method`` is the HTTP verb the inbound
    request used; the dispatcher forwards it verbatim under passthrough.
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
        """Concrete path the dispatcher forwards to upstream."""
        return self.actual_path if self.actual_path is not None else self.endpoint
