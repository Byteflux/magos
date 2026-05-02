"""Run magos as a single-process FastAPI server, or invoke CLI subcommands.

Default invocation starts the server::

    magos                          # serve
    magos serve
    magos serve --config /path/to/magos.yaml

Operator-facing subcommands::

    magos models list
    magos models show <id>
    magos models refresh [--provider X]
    magos models prune
    magos models discover --provider X --dry-run

The ``magos`` script is installed by the ``[project.scripts]`` entry in
``pyproject.toml``. Inside a uv-managed venv use ``uv run magos …``;
``python -m magos`` works as an alternative invocation.

Config resolution order (highest first):

1. ``--config`` CLI flag (any subcommand)
2. ``MAGOS_CONFIG_PATH`` env var
3. ``~/.magos/magos.yaml`` (default)

All other knobs live in ``MagosSettings`` (see ``magos.config``); set
them via environment variables prefixed ``MAGOS_`` or a local ``.env``.
"""

from __future__ import annotations

import os
import sys

import uvicorn

from magos.cli.models_cmd import main as models_main
from magos.config import MagosSettings
from magos.obs import configure_logging, configure_tracing


def _consume_config_flag(argv: list[str]) -> list[str]:
    """Strip ``--config <path>`` (or ``--config=<path>``) from ``argv``.

    Sets ``MAGOS_CONFIG_PATH`` so downstream ``MagosSettings()`` picks
    it up via env. Returns the remaining args. Lets every subcommand
    accept ``--config`` uniformly without each one re-implementing it.
    """
    out: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--config" and i + 1 < len(argv):
            os.environ["MAGOS_CONFIG_PATH"] = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--config="):
            os.environ["MAGOS_CONFIG_PATH"] = arg.split("=", 1)[1]
            i += 1
            continue
        out.append(arg)
        i += 1
    return out


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
    args = _consume_config_flag(args)
    if not args or args[0] == "serve":
        serve()
        return 0
    if args[0] == "models":
        return models_main(args[1:])
    print(f"unknown subcommand: {args[0]!r}; expected 'serve' or 'models'", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
