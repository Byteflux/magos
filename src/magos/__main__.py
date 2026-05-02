"""Run magos as a single-process FastAPI server, or invoke CLI subcommands.

Default invocation starts the server::

    python -m magos
    python -m magos serve

Operator-facing subcommands::

    python -m magos models list
    python -m magos models show <id>
    python -m magos models refresh [--provider X]
    python -m magos models prune
    python -m magos models discover --provider X --dry-run

All knobs live in ``MagosSettings`` (see ``magos.config``); set them via
environment variables prefixed ``MAGOS_`` or a local ``.env`` file.
"""

from __future__ import annotations

import sys

import uvicorn

from magos.cli.models_cmd import main as models_main
from magos.config import MagosSettings
from magos.obs import configure_logging, configure_tracing


def serve() -> None:
    settings = MagosSettings()
    configure_logging(level=settings.log_level, json=settings.log_json)
    configure_tracing(endpoint=settings.otel_endpoint, enabled=settings.otel_enabled)
    uvicorn.run(
        "magos.server:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_config=None,
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] == "serve":
        serve()
        return 0
    if args[0] == "models":
        return models_main(args[1:])
    print(f"unknown subcommand: {args[0]!r}; expected 'serve' or 'models'", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
