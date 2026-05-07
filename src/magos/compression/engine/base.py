"""Marker base class for all compression engine steps.

`Compressor` extends `Transform` to signal that a transform may consult
the registry for model-limit resolution. Provider and endpoint dispatch
lives inside the concrete `Compress` schema's `apply` method.
"""

from __future__ import annotations

from abc import ABC

from magos.routing.transform import Transform


class Compressor(Transform, ABC):
    """Marker subclass. Compressors may consult the registry for model_limit
    resolution; provider/endpoint dispatch lives inside `Compress`."""
