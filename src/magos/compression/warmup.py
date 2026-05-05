"""Walk a ``PipelineRegistry`` and eagerly load each unique transform.

Mirrors the proxy's startup behaviour (``proxy/server.py``): dedupe by
``id()`` so two pipelines sharing a transform instance pay the cost
once, swallow per-transform errors so a single failure cannot break
process startup.
"""

from __future__ import annotations

from magos.telemetry import get_logger

from .registry import PipelineRegistry, get_registry

log = get_logger("magos.compression")


def eager_warmup(registry: PipelineRegistry | None = None) -> None:
    """Call ``eager_load_compressors`` on each unique transform."""
    reg = registry if registry is not None else get_registry()
    seen: set[int] = set()
    for pipeline in reg.pipelines():
        for transform in getattr(pipeline, "transforms", []):
            if id(transform) in seen:
                continue
            seen.add(id(transform))
            loader = getattr(transform, "eager_load_compressors", None)
            if loader is None:
                continue
            try:
                loader()
            except Exception as exc:
                log.warning(
                    "compress.eager_load_failed",
                    transform=type(transform).__name__,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
