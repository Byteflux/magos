"""Smoke tests for observability scaffolding.

These don't try to assert on emitted spans or log records; they verify the
no-op contract: configure_* is idempotent, traced preserves behavior, and
get_logger never raises.
"""

from __future__ import annotations

import pytest

from magos.obs import configure_logging, configure_tracing, get_logger, traced


@pytest.mark.unit
def test_traced_preserves_return_value() -> None:
    @traced()
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5


@pytest.mark.unit
def test_traced_propagates_exceptions() -> None:
    @traced("failing-op")
    def boom() -> None:
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        boom()


@pytest.mark.unit
def test_get_logger_emits_without_raising() -> None:
    log = get_logger("magos.test")
    log.info("ping", key="value")


@pytest.mark.unit
def test_configure_tracing_is_noop_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MAGOS_OTEL_ENABLED", raising=False)
    configure_tracing()


@pytest.mark.unit
def test_configure_logging_is_idempotent() -> None:
    configure_logging()
    configure_logging(json=True)
    get_logger().info("ok")
