"""Run magos as a single-process FastAPI server.

Usage::

    python -m magos

All knobs live in ``MagosSettings`` (see ``magos.config``); set them via
environment variables prefixed ``MAGOS_`` or a local ``.env`` file.
"""

from __future__ import annotations

import uvicorn

from magos.config import MagosSettings
from magos.obs import configure_logging, configure_tracing


def main() -> None:
    settings = MagosSettings()
    configure_logging(level=settings.log_level, json=settings.log_json)
    configure_tracing(endpoint=settings.otel_endpoint, enabled=settings.otel_enabled)
    uvicorn.run(
        "magos.server:app",
        host=settings.host,
        port=settings.port,
        log_config=None,
    )


if __name__ == "__main__":
    main()
