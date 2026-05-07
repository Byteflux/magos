"""Pure mutators for the routing pipeline. See `docs/routing/grammar.md`.

Each transform returns a new `RoutedRequest`; body-touching ops flip
`body_dirty` so passthrough re-serialises. Every transform implements
`Transform.apply`, so dispatch is fully polymorphic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from magos.routing.rewrites.jq_patch import RewriteError

if TYPE_CHECKING:
    from magos.registry.state import RegistryState
    from magos.routing.request import RoutedRequest

__all__ = ["RewriteError", "apply_transforms"]


def apply_transforms(
    req: RoutedRequest,
    transforms: Sequence[Any],
    *,
    registry: RegistryState | None = None,
) -> RoutedRequest:
    """Apply `transforms` in list order; return a new `RoutedRequest`.

    Empty list returns `req` unchanged. `registry` is forwarded to the
    compress transform for context-size resolution.
    """
    if not transforms:
        return req
    out = req
    for rw in transforms:
        out = rw.apply(out, registry=registry)
    return out
