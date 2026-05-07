"""Smoke tests for `magos.telemetry.logging`.

These verify the no-op contract: `configure_logging` is idempotent
and `get_logger` never raises.
"""

from __future__ import annotations

import pytest

from magos.telemetry import configure_logging, get_logger


@pytest.mark.unit
def test_get_logger_emits_without_raising() -> None:
    log = get_logger("magos.test")
    log.info("ping", key="value")


@pytest.mark.unit
def test_configure_logging_is_idempotent() -> None:
    configure_logging()
    configure_logging(json=True)
    get_logger().info("ok")
