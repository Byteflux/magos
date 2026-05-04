"""Forward mitmproxy's stdlib-``logging`` records into structlog.

mitmproxy 12 emits its operational events through the standard
``logging`` framework (``logging.getLogger("mitmproxy")`` and
descendants). Without a bridge, magos's structlog setup and
mitmproxy's plain log records produce two interleaved formats. The
handler installed here re-emits each record via
``magos.telemetry.get_logger`` so the unified line shape (timestamp, level,
event, key/value pairs) holds across both servers.

Idempotent: re-installing replaces handlers rather than appending,
so callers don't need to track installation state.
"""

from __future__ import annotations

import logging

from magos.telemetry import get_logger


class StructlogHandler(logging.Handler):
    """Re-emit a stdlib ``LogRecord`` via the named structlog logger."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:
            # ``getMessage`` can fail if the format args are malformed;
            # falling through to ``record.msg`` keeps the bridge robust
            # regardless of how mitmproxy formats internal events.
            message = str(record.msg)
        log = get_logger(record.name)
        method = getattr(log, record.levelname.lower(), log.info)
        method(message)


def install_log_bridge() -> None:
    """Route mitmproxy's logger through structlog.

    Replaces any handlers already attached to the ``mitmproxy`` logger
    (mitmproxy installs its own at startup) and disables propagation so
    the root logger's handlers don't double-emit.
    """
    bridge = StructlogHandler()
    mitm_logger = logging.getLogger("mitmproxy")
    mitm_logger.handlers = [bridge]
    mitm_logger.propagate = False
