"""OTel metrics + structlog helpers for the registry.

Instruments bind to the OTel global meter at import time. See
``docs/registry/observability.md``.
"""

from __future__ import annotations

from collections.abc import Iterable

from opentelemetry import metrics

from magos.telemetry import get_logger

log = get_logger("magos.registry")

_METER_NAME = "magos.registry"

_meter = metrics.get_meter(_METER_NAME)

# Counters
_refresh_total = _meter.create_counter(
    "magos.registry.refresh.total",
    description="Refresh attempts grouped by provider and status",
)
_refresh_failure_total = _meter.create_counter(
    "magos.registry.refresh.failures",
    description="Refresh failures by provider and reason",
)
_models_added_total = _meter.create_counter(
    "magos.registry.models.added",
    description="Models newly registered (first appearance) by provider",
)
_models_deprecated_total = _meter.create_counter(
    "magos.registry.models.deprecated",
    description="Models marked deprecated_at this refresh by provider",
)
_models_pruned_total = _meter.create_counter(
    "magos.registry.models.pruned",
    description="Models hard-deleted past grace by provider",
)

# Histogram
_refresh_duration = _meter.create_histogram(
    "magos.registry.refresh.duration",
    unit="s",
    description="Wall time per refresh attempt by provider",
)

# Per-provider gauge tracked via observable; updated by setting cached snapshots.
_models_total_snapshot: dict[str, int] = {}


def _models_total_callback(
    options: metrics.CallbackOptions,
) -> Iterable[metrics.Observation]:
    """Emit one observation per provider on each scrape."""
    for provider, count in _models_total_snapshot.items():
        yield metrics.Observation(value=count, attributes={"provider": provider})


_models_total = _meter.create_observable_gauge(
    "magos.registry.models.total",
    callbacks=[_models_total_callback],
    description="Active models in the registry by provider (excludes pruned)",
)


def record_refresh_attempt(provider: str) -> None:
    _refresh_total.add(1, {"provider": provider, "status": "attempt"})
    log.debug("registry.refresh.attempt", provider=provider)


def record_refresh_success(
    provider: str,
    *,
    duration_seconds: float,
    total: int,
    added: int,
    deprecated: int,
    pruned: int,
) -> None:
    _refresh_total.add(1, {"provider": provider, "status": "success"})
    _refresh_duration.record(duration_seconds, {"provider": provider})
    if added:
        _models_added_total.add(added, {"provider": provider})
    if deprecated:
        _models_deprecated_total.add(deprecated, {"provider": provider})
    if pruned:
        _models_pruned_total.add(pruned, {"provider": provider})
    _models_total_snapshot[provider] = total
    log.info(
        "registry.refresh.success",
        provider=provider,
        duration_seconds=round(duration_seconds, 4),
        total=total,
        added=added,
        deprecated=deprecated,
        pruned=pruned,
    )


def record_refresh_failure(provider: str, *, duration_seconds: float, error: BaseException) -> None:
    _refresh_total.add(1, {"provider": provider, "status": "failure"})
    _refresh_failure_total.add(1, {"provider": provider, "error_type": type(error).__name__})
    _refresh_duration.record(duration_seconds, {"provider": provider})
    log.warning(
        "registry.refresh.failure",
        provider=provider,
        duration_seconds=round(duration_seconds, 4),
        error=str(error),
        error_type=type(error).__name__,
    )


def reset_for_tests() -> None:
    """Clear cached gauge snapshots; intended for unit tests only."""
    _models_total_snapshot.clear()


__all__ = [
    "record_refresh_attempt",
    "record_refresh_failure",
    "record_refresh_success",
    "reset_for_tests",
]
