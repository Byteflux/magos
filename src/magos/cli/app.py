"""Root Typer app: top-level options and subcommands.

Subcommands::

    magos serve                    # run the FastAPI server
    magos serve --config x.yaml    # with a non-default config
    magos models list
    magos models show <id>
    magos models refresh [--provider X]
    magos models prune
    magos models discover --provider X [--dry-run / --no-dry-run]

Invoking ``magos`` with no subcommand prints help; ``serve`` is required
to start the server.

The ``magos`` script is installed by the ``[project.scripts]`` entry in
``pyproject.toml``. Inside a uv-managed venv use ``uv run magos …``;
``python -m magos`` works as an alternative invocation.

Config resolution order (highest first):

1. ``--config`` CLI flag (top-level option, before the subcommand)
2. ``MAGOS_CONFIG_PATH`` env var
3. ``~/.magos/magos.yaml`` (default)

All other knobs live in ``MagosSettings`` (see :mod:`magos.config`); set
them via environment variables prefixed ``MAGOS_`` or a local ``.env``.
"""

from __future__ import annotations

import os
from typing import Annotated

import typer

from magos import __version__
from magos.cli import models, serve

app = typer.Typer(
    name="magos",
    help="LLM proxy server with provider-discovered model registry.",
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
app.add_typer(models.models_app, name="models")
serve.register(app)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"magos {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def _root(
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


def main() -> None:
    """Console-script entrypoint."""
    app()
