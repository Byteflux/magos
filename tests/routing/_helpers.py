"""Shared helpers for routing tests.

``make_req`` is the canonical builder for ``RoutedRequest`` instances in
unit tests. Use it everywhere instead of constructing the dataclass
directly so default fields stay consistent across the routing test
suite.

``make_registry`` wraps ``ModelEntry`` instances in a ``RegistryState``
so tests can hand a populated registry to ``route()`` / matchers.
"""

from __future__ import annotations

from typing import Any

from magos.registry.state import ModelEntry, RegistryState
from magos.routing import RoutedRequest
from magos.routing.request import Endpoint


def make_req(
    *,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    raw: bytes = b"",
    body_dirty: bool = False,
    endpoint: Endpoint = "/v1/messages",
) -> RoutedRequest:
    return RoutedRequest(
        endpoint=endpoint,
        headers=headers or {},
        body=body or {},
        raw_body=raw,
        body_dirty=body_dirty,
    )


def make_registry(*entries: ModelEntry) -> RegistryState:
    """Build a ``RegistryState`` keyed by ``namespaced_id``."""
    return RegistryState(entries={e.namespaced_id: e for e in entries})
