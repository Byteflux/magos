"""Pre-build pipelines for every routing-config Compress and warm transforms.

Two functions:

- ``eager_warmup`` walks a ``PipelineRegistry`` and eagerly loads each
  unique transform. Dedupes by ``id()`` so shared transforms pay the
  load cost once.
- ``prebuild_from_routing`` walks a ``RoutingConfig``, builds every
  (config, provider) pipeline implied by token-mode ``Compress``
  rewrites, then calls ``eager_warmup``. Cache-mode Compress is
  skipped (no pipeline involved). Per-pipeline failures are logged
  and skipped so one bad config doesn't break startup.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from magos.telemetry import get_logger

from .config import pipeline_config_from_compress_options
from .registry import PipelineRegistry, get_registry

if TYPE_CHECKING:
    from magos.routing.schema import CompressOptions, RoutingConfig

log = get_logger("magos.compression")

_PREBUILD_PROVIDERS: tuple[str, ...] = ("anthropic", "openai")


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


def prebuild_from_routing(cfg: RoutingConfig, registry: PipelineRegistry | None = None) -> None:
    """Build every (PipelineConfig, provider) pipeline implied by ``cfg``.

    Walks ``cfg.pre_rewrites`` (including ``GuardedRewrites``) and each
    rule's ``rewrites``; for every token-mode ``Compress``, transcodes
    the options to a ``PipelineConfig`` and calls ``registry.get_or_build``
    for both providers. Calls ``eager_warmup(registry)`` at the end so
    transform models are loaded for the freshly-built pipelines.

    Per-pipeline build errors are logged as ``compress.pipeline_prebuild_failed``
    and skipped; the walk continues. The registry's fingerprint dedup
    means duplicate configs are O(1).
    """
    reg = registry if registry is not None else get_registry()

    seen_keys: set[tuple[str, str]] = set()
    n_pipelines = 0
    for opts in _iter_token_mode_compress_options(cfg):
        pc = pipeline_config_from_compress_options(opts)
        for provider in _PREBUILD_PROVIDERS:
            key = (pc.fingerprint(), provider)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            try:
                reg.get_or_build(pc, provider_name=provider)  # type: ignore[arg-type]
                n_pipelines += 1
            except Exception as exc:
                log.warning(
                    "compress.pipeline_prebuild_failed",
                    fingerprint=pc.fingerprint(),
                    provider=provider,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )

    n_configs = len({fp for fp, _ in seen_keys})
    log.info("compress.pipeline_prebuilt", n_configs=n_configs, n_pipelines=n_pipelines)
    eager_warmup(reg)


def _iter_token_mode_compress_options(cfg: RoutingConfig) -> Iterator[CompressOptions]:
    """Yield each token-mode ``CompressOptions`` from pre_rewrites + rules."""
    from magos.routing.schema import Compress, GuardedRewrites  # noqa: PLC0415

    for entry in cfg.pre_rewrites:
        if isinstance(entry, Compress):
            if entry.compress.mode == "token":
                yield entry.compress
        elif isinstance(entry, GuardedRewrites):
            for inner in entry.rewrites:
                if isinstance(inner, Compress) and inner.compress.mode == "token":
                    yield inner.compress
    for rule in cfg.rules:
        for rw in rule.rewrites:
            if isinstance(rw, Compress) and rw.compress.mode == "token":
                yield rw.compress
