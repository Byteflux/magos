"""Run magos as a single-process FastAPI server, or invoke CLI subcommands.

Default invocation starts the server::

    magos                          # serve (no subcommand)
    magos serve                    # explicit
    magos --config /etc/x.yaml     # config override

Operator-facing subcommands::

    magos models list
    magos models show <id>
    magos models refresh [--provider X]
    magos models prune
    magos models discover --provider X [--dry-run / --no-dry-run]

The ``magos`` script is installed by the ``[project.scripts]`` entry in
``pyproject.toml``. Inside a uv-managed venv use ``uv run magos …``;
``python -m magos`` works as an alternative invocation.

Config resolution order (highest first):

1. ``--config`` CLI flag (top-level option, before the subcommand)
2. ``MAGOS_CONFIG_PATH`` env var
3. ``~/.magos/magos.yaml`` (default)

All other knobs live in ``MagosSettings`` (see ``magos.config``); set
them via environment variables prefixed ``MAGOS_`` or a local ``.env``.
"""

from __future__ import annotations

import os
from typing import Annotated

import typer

from magos import __version__
from magos.cli.models_cmd import models_app
from magos.config.settings import MagosSettings
from magos.serve import serve as serve_orchestrator
from magos.telemetry import configure_logging, configure_tracing, get_logger

app = typer.Typer(
    name="magos",
    help="LLM proxy server with provider-discovered model registry.",
    no_args_is_help=False,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
app.add_typer(models_app, name="models")


def serve(host: str | None = None, port: int | None = None) -> None:
    """Boot the FastAPI server (and optional mitm ingress) under one process.

    ``host`` and ``port`` override the values resolved from the environment
    (``MAGOS_HOST`` / ``MAGOS_PORT``); env in turn overrides ``server.host``
    / ``server.port`` from ``magos.yaml``. The mitm ingress is started
    alongside FastAPI when ``server.ingress.enabled`` is true in yaml — see
    ``docs/ingress.md`` for setup.
    """
    if host is not None:
        os.environ["MAGOS_HOST"] = host
    if port is not None:
        os.environ["MAGOS_PORT"] = str(port)
    settings = MagosSettings()
    configure_logging(level=settings.log_level, json=settings.log_json)
    configure_tracing(endpoint=settings.otel_endpoint, enabled=settings.otel_enabled)
    log = get_logger("magos")
    log.info(
        "server.bootstrapping",
        version=__version__,
        config_path=settings.config_path,
        models_path_override=settings.models_path,
        log_level=settings.log_level,
        log_json=settings.log_json,
        otel_enabled=settings.otel_enabled,
        metrics_enabled=settings.metrics_enabled,
        access_log=settings.access_log,
        kompress_backend=settings.kompress_backend,
    )
    serve_orchestrator(settings=settings)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"magos {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    config: Annotated[
        str | None,
        typer.Option(
            "--config",
            help="Path to magos.yaml (overrides MAGOS_CONFIG_PATH and the ~/.magos/magos.yaml default).",
        ),
    ] = None,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Print the magos version and exit.",
        ),
    ] = False,
) -> None:
    if config is not None:
        os.environ["MAGOS_CONFIG_PATH"] = config
    if ctx.invoked_subcommand is None:
        serve()


@app.command("serve")
def serve_cmd(
    host: Annotated[
        str | None,
        typer.Option(
            "--host",
            help="HTTP listen host (overrides MAGOS_HOST; default 127.0.0.1).",
        ),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option(
            "--port",
            "-p",
            min=1,
            max=65535,
            help="HTTP listen port (overrides MAGOS_PORT; default 8000).",
        ),
    ] = None,
) -> None:
    """Run the FastAPI server (the default when no subcommand is given)."""
    serve(host=host, port=port)


def main() -> None:
    """Console-script entrypoint."""
    app()


if __name__ == "__main__":
    main()
