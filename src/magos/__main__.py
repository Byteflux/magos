"""Run magos as a single-process FastAPI server.

Usage::

    python -m magos
"""

from __future__ import annotations

import os

import uvicorn

from magos.obs import configure_logging, configure_tracing

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def main() -> None:
    configure_logging(level=os.environ.get("MAGOS_LOG_LEVEL", "INFO"))
    configure_tracing()
    uvicorn.run(
        "magos.server:app",
        host=os.environ.get("MAGOS_HOST", DEFAULT_HOST),
        port=int(os.environ.get("MAGOS_PORT", DEFAULT_PORT)),
        log_config=None,
    )


if __name__ == "__main__":
    main()
