"""Tests for the structlog bridge that forwards mitmproxy log records."""

from __future__ import annotations

import logging

import pytest
import structlog

from magos.ingress.mitm.log_bridge import StructlogHandler, install_log_bridge


@pytest.mark.unit
def test_handler_emits_each_record_via_structlog() -> None:
    handler = StructlogHandler()
    record = logging.LogRecord(
        name="mitmproxy.proxy.server",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Proxy server listening at *:8080",
        args=None,
        exc_info=None,
    )
    with structlog.testing.capture_logs() as logs:
        handler.emit(record)
    assert len(logs) == 1
    assert logs[0]["event"] == "Proxy server listening at *:8080"
    assert logs[0]["log_level"] == "info"


@pytest.mark.unit
def test_handler_routes_warning_records() -> None:
    handler = StructlogHandler()
    record = logging.LogRecord(
        name="mitmproxy.tls",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="TLS handshake failed",
        args=None,
        exc_info=None,
    )
    with structlog.testing.capture_logs() as logs:
        handler.emit(record)
    assert logs[0]["log_level"] == "warning"


@pytest.mark.unit
def test_handler_falls_back_to_raw_msg_on_format_error() -> None:
    handler = StructlogHandler()
    # %s arg count mismatch — getMessage will raise.
    record = logging.LogRecord(
        name="mitmproxy",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="bad %s %s",
        args=("only-one",),
        exc_info=None,
    )
    with structlog.testing.capture_logs() as logs:
        handler.emit(record)
    # Doesn't crash and emits something (the raw msg).
    assert len(logs) == 1


@pytest.mark.unit
def test_install_log_bridge_replaces_existing_handlers() -> None:
    mitm_logger = logging.getLogger("mitmproxy")
    sentinel = logging.NullHandler()
    mitm_logger.handlers = [sentinel]
    mitm_logger.propagate = True
    try:
        install_log_bridge()
        assert len(mitm_logger.handlers) == 1
        assert isinstance(mitm_logger.handlers[0], StructlogHandler)
        assert mitm_logger.propagate is False
    finally:
        mitm_logger.handlers = []
        mitm_logger.propagate = True
