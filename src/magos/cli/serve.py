"""``magos serve`` command and the entrypoint bootstrap.

The bootstrap stamps CLI ``--host`` / ``--port`` flags into env vars so
the same env-over-yaml resolution used by the orchestrator picks them
up, configures structlog + OTel from ``MagosSettings``, emits the
``server.bootstrapping`` log event, then hands off to
:func:`magos.serve.serve` (the sync wrapper around the async
orchestrator).

The bootstrap lives here rather than in :mod:`magos.serve` because
logging/tracing configuration and the bootstrap log event are
entrypoint concerns: a library caller of ``magos.serve.serve_async``
should not be implicitly reconfiguring the root logger.
"""

from __future__ import annotations

import os
from typing import Annotated

import typer

from magos import __version__
from magos.config.settings import MagosSettings
from magos.serve import serve as serve_orchestrator
from magos.telemetry import configure_logging, configure_tracing, get_logger


def bootstrap_and_serve(host: str | None = None, port: int | None = None) -> None:
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


def register(app: typer.Typer) -> None:
    """Attach the ``serve`` command to the root Typer app."""

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
        bootstrap_and_serve(host=host, port=port)
