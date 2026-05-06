"""Request abstraction shared by routing and dispatch. See ``docs/routing/pipeline.md``."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from magos.egress.usage import Usage

PostResponseHook = Callable[["Usage"], None]
"""Closure fired by egress dispatch after the upstream's ``Usage`` is captured.

Used by the compress rewrite (Phase 1.5) to feed cache_read / cache_write
tokens back into the per-session ``PrefixCacheTracker``. Hooks should not
raise; dispatch swallows + logs failures so one bad hook can't break the
client response.
"""

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
    on templated paths. ``post_response_hooks`` is a mutable list of closures
    appended by rewrites and fired by dispatch after the upstream response's
    ``Usage`` is captured.
    """

    endpoint: Endpoint
    headers: Mapping[str, str]
    body: Mapping[str, Any]
    raw_body: bytes
    body_dirty: bool = False
    method: HttpMethod = "POST"
    actual_path: str | None = None
    post_response_hooks: list[PostResponseHook] = field(default_factory=list)

    @property
    def forward_path(self) -> str:
        return self.actual_path if self.actual_path is not None else self.endpoint
