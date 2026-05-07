"""Marker base class for all rewrite primitives.

`Rewriter` extends `Transform` to signal that a transform operates
synchronously and does not need the registry. The distinction is
semantic; the `registry` kwarg is still accepted (and ignored) so all
transforms share the same `apply` signature.
"""

from __future__ import annotations

from abc import ABC

from magos.routing.transform import Transform


class Rewriter(Transform, ABC):
    """Marker subclass. Rewriters mutate `RoutedRequest` synchronously
    and don't need the registry."""
