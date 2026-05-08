"""Smoke tests for `magos.telemetry.logging`.

These verify the no-op contract (`configure_logging` is idempotent,
`get_logger` never raises) plus the level-routing baseline: the root
logger sits at the third-party floor (default ERROR) and `magos.*` is
bumped to the operator-configured level.
"""

from __future__ import annotations

import io
import logging
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


@pytest.fixture
def reset_logging() -> object:
    """Snapshot+restore root and the loggers configure_logging mutates.

    `configure_logging` rewrites root handlers + levels and a few named
    loggers (uvicorn, magos, LiteLLM); without this fixture, level-checking
    tests pollute later tests.
    """
    names = ("", "magos", "uvicorn", "uvicorn.error", "uvicorn.access", "LiteLLM")
    snapshot = [(n, logging.getLogger(n).level, list(logging.getLogger(n).handlers)) for n in names]
    yield None
    for name, level, handlers in snapshot:
        lg = logging.getLogger(name)
        lg.setLevel(level)
        lg.handlers = handlers


@pytest.mark.unit
def test_root_floor_defaults_to_error_and_magos_is_info(
    monkeypatch: pytest.MonkeyPatch, reset_logging: object
) -> None:
    monkeypatch.delenv("MAGOS_THIRD_PARTY_LOG_LEVEL", raising=False)
    configure_logging()
    assert logging.getLogger().level == logging.ERROR
    assert logging.getLogger("magos").level == logging.INFO


@pytest.mark.unit
def test_third_party_floor_env_override(
    monkeypatch: pytest.MonkeyPatch, reset_logging: object
) -> None:
    monkeypatch.setenv("MAGOS_THIRD_PARTY_LOG_LEVEL", "WARNING")
    configure_logging()
    assert logging.getLogger().level == logging.WARNING


@pytest.mark.unit
def test_third_party_loggers_silent_at_info(
    monkeypatch: pytest.MonkeyPatch, reset_logging: object, capsys: pytest.CaptureFixture[str]
) -> None:
    """A library logging at INFO is dropped at the root floor (ERROR)."""
    monkeypatch.delenv("MAGOS_THIRD_PARTY_LOG_LEVEL", raising=False)
    configure_logging()

    logging.getLogger("uvicorn.access").info("HTTP request that should NOT show up")
    logging.getLogger("LiteLLM").info("chatty LiteLLM noise")

    captured = capsys.readouterr()
    assert "HTTP request that should NOT show up" not in captured.err
    assert "chatty LiteLLM noise" not in captured.err


@pytest.mark.unit
def test_magos_logger_level_follows_level_arg(reset_logging: object) -> None:
    configure_logging(level="DEBUG")
    assert logging.getLogger("magos").level == logging.DEBUG
