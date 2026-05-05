"""Pure mutators for the routing pipeline. See ``docs/routing/grammar.md``.

Each rewrite returns a new ``RoutedRequest``; body-touching ops flip
``body_dirty`` so passthrough re-serialises. Per-primitive logic lives in
sibling modules; ``apply_rewrites`` is the dispatch entry point.
"""

from __future__ import annotations

from collections.abc import Sequence

from magos.registry.state import RegistryState
from magos.routing.request import RoutedRequest
from magos.routing.rewrites.compress import apply_compress
from magos.routing.rewrites.headers import (
    apply_add_header,
    apply_remove_header,
    apply_set_header,
)
from magos.routing.rewrites.jq_patch import RewriteError, apply_jq_patch
from magos.routing.rewrites.model import apply_set_model
from magos.routing.schema import (
    AddHeader,
    Compress,
    JqPatch,
    RemoveHeader,
    Rewrite,
    SetHeader,
    SetModel,
)

__all__ = ["RewriteError", "apply_rewrites"]


def apply_rewrites(
    req: RoutedRequest,
    rewrites: Sequence[Rewrite],
    *,
    registry: RegistryState | None = None,
) -> RoutedRequest:
    """Apply ``rewrites`` in list order; return a new ``RoutedRequest``.

    Empty list returns ``req`` unchanged. ``registry`` is forwarded to the
    compress rewrite for context-size resolution.
    """
    if not rewrites:
        return req
    out = req
    for rw in rewrites:
        out = _apply_one(out, rw, registry=registry)
    return out


def _apply_one(
    req: RoutedRequest, rw: Rewrite, *, registry: RegistryState | None = None
) -> RoutedRequest:
    if isinstance(rw, SetModel):
        return apply_set_model(req, rw)
    if isinstance(rw, SetHeader):
        return apply_set_header(req, rw)
    if isinstance(rw, AddHeader):
        return apply_add_header(req, rw)
    if isinstance(rw, RemoveHeader):
        return apply_remove_header(req, rw)
    if isinstance(rw, JqPatch):
        return apply_jq_patch(req, rw)
    if isinstance(rw, Compress):
        return apply_compress(req, rw, registry=registry)
    raise TypeError(f"unhandled Rewrite variant: {type(rw).__name__}")
