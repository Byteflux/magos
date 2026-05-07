"""Smoke tests for `magos.telemetry.logging`.

These verify the no-op contract: `configure_logging` is idempotent
and `get_logger` never raises.
"""

from __future__ import annotations

import io
import sys

import pytest
import structlog

from magos.telemetry import configure_logging, get_logger
from magos.telemetry.logging import _exception_formatter


@pytest.mark.unit
def test_get_logger_emits_without_raising() -> None:
    log = get_logger("magos.test")
    log.info("ping", key="value")


@pytest.mark.unit
def test_configure_logging_is_idempotent() -> None:
    configure_logging()
    configure_logging(json=True)
    get_logger().info("ok")


@pytest.mark.unit
def test_exception_formatter_hides_locals() -> None:
    """Locals are the loud failure mode (request payloads, API keys); ensure they're hidden."""
    fmt = _exception_formatter()
    if isinstance(fmt, structlog.dev.RichTracebackFormatter):
        assert fmt.show_locals is False
    else:
        # plain_traceback fallback is fine; it never emits locals.
        assert fmt is structlog.dev.plain_traceback


@pytest.mark.unit
def test_exception_formatter_render_omits_locals() -> None:
    """Render an exception through the formatter and confirm no locals dump."""
    secret = "sk-do-not-leak-this"
    try:
        # Bind `secret` as a local so a `show_locals=True` formatter would
        # surface it; the assertion below proves it doesn't.
        assert secret
        raise RuntimeError("boom")
    except RuntimeError:
        sio = io.StringIO()
        _exception_formatter()(sio, sys.exc_info())
        output = sio.getvalue()

    assert "boom" in output
    assert secret not in output
