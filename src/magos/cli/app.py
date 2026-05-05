"""Root Typer app: top-level options and subcommands. See ``docs/cli.md``."""

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
    home: Annotated[
        str | None,
        typer.Option(
            "--home",
            help="Path to the magos data directory (overrides MAGOS_HOME; default ~/.magos).",
        ),
    ] = None,
    config: Annotated[
        str | None,
        typer.Option(
            "--config",
            help="Path to magos.yaml (overrides MAGOS_CONFIG_PATH and the $MAGOS_HOME/magos.yaml default).",
        ),
    ] = None,
    models: Annotated[
        str | None,
        typer.Option(
            "--models",
            help=(
                "Path to models.json (overrides MAGOS_MODELS_PATH and the "
                "yaml ``registry.models_path``; default $MAGOS_HOME/models.json)."
            ),
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
    if home is not None:
        os.environ["MAGOS_HOME"] = home
    if config is not None:
        os.environ["MAGOS_CONFIG_PATH"] = config
    if models is not None:
        os.environ["MAGOS_MODELS_PATH"] = models


def main() -> None:
    app()
