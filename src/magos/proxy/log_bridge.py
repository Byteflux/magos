"""Forward mitmproxy 12's stdlib `logging` records into structlog so
both servers share one line shape. Re-emits under `magos.proxy`
(not `mitmproxy.*`); structlog's stdlib `LoggerFactory`
writes back under the bound name, so reusing `mitmproxy.*` would feed
records into this handler and recurse unboundedly on startup. Idempotent:
re-installing replaces handlers."""

from __future__ import annotations

import logging

from magos.telemetry import get_logger

_BRIDGE_LOGGER = "magos.proxy"


class StructlogHandler(logging.Handler):
    """Re-emit a stdlib `LogRecord` via structlog under a magos namespace."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:
            # `getMessage` can fail on malformed format args; fall
            # back to raw msg so the bridge stays robust.
            message = str(record.msg)
        log = get_logger(_BRIDGE_LOGGER)
        method = getattr(log, record.levelname.lower(), log.info)
        method(message, logger=record.name)


def install_log_bridge() -> None:
    """Route `mitmproxy` logger through structlog. Replaces existing
    handlers and disables propagation to avoid double-emit."""
    bridge = StructlogHandler()
    mitm_logger = logging.getLogger("mitmproxy")
    mitm_logger.handlers = [bridge]
    mitm_logger.propagate = False
