"""``magos serve`` command + entrypoint bootstrap (logging/tracing config + bootstrap log).

Bootstrap lives here rather than in :mod:`magos.serve` so library callers
of ``serve_async`` don't implicitly reconfigure the root logger.
"""

from __future__ import annotations

import os
from typing import Annotated

import typer

from magos import __version__
from magos.config.settings import MagosSettings
from magos.telemetry import configure_logging, configure_tracing, get_logger


def bootstrap_and_serve(
    host: str | None = None,
    port: int | None = None,
    enable_mitm: bool | None = None,
    mitm_host: str | None = None,
    mitm_port: int | None = None,
) -> None:
    """Boot the FastAPI server (and optional mitm ingress); CLI > env > yaml. See ``docs/cli.md``."""
    if host is not None:
        os.environ["MAGOS_HOST"] = host
    if port is not None:
        os.environ["MAGOS_PORT"] = str(port)
    if enable_mitm is not None:
        os.environ["MAGOS_MITM_ENABLED"] = "1" if enable_mitm else "0"
    if mitm_host is not None:
        os.environ["MAGOS_MITM_HOST"] = mitm_host
    if mitm_port is not None:
        os.environ["MAGOS_MITM_PORT"] = str(mitm_port)
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
    _preload_native_load_order(settings)
    from magos.serve import serve as serve_orchestrator  # noqa: PLC0415

    serve_orchestrator(settings=settings)


def _preload_native_load_order(settings: MagosSettings) -> None:
    """Force ``sentence_transformers`` import before any PyO3-bound module.

    Wins the Windows native-load-order race (segfault in pyarrow's
    ``.pyd`` after PyO3 init); see ``docs/headroom/pipeline.md``.
    Doing it here, before ``magos.serve`` imports litellm, makes the
    order deterministic. Skipped when no compress rule is configured,
    since the import is multi-second.
    """
    from magos.config.loader import load_full_config  # noqa: PLC0415
    from magos.routing import config_uses_compress  # noqa: PLC0415

    try:
        cfg = load_full_config(settings.config_path).routing
    except Exception:
        return  # malformed config; let serve_orchestrator surface the error

    if not config_uses_compress(cfg):
        return

    from magos.routing.rewrites.compress import (  # noqa: PLC0415
        _preload_sentence_transformers,
    )

    _preload_sentence_transformers()
    get_logger("magos").debug("compress.sentence_transformers_preloaded")


def register(app: typer.Typer) -> None:
    """Attach ``serve`` to the root Typer app."""

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
                help="HTTP listen port (overrides MAGOS_PORT; default 6246).",
            ),
        ] = None,
        enable_mitm: Annotated[
            bool | None,
            typer.Option(
                "--enable-mitm/--disable-mitm",
                help=(
                    "Toggle the embedded mitmproxy HTTPS_PROXY listener "
                    "(overrides MAGOS_MITM_ENABLED and ingress.mitm.enabled)."
                ),
            ),
        ] = None,
        mitm_host: Annotated[
            str | None,
            typer.Option(
                "--mitm-host",
                help=("mitmproxy listener host (overrides MAGOS_MITM_HOST; default 127.0.0.1)."),
            ),
        ] = None,
        mitm_port: Annotated[
            int | None,
            typer.Option(
                "--mitm-port",
                min=1,
                max=65535,
                help=("mitmproxy listener port (overrides MAGOS_MITM_PORT; default 6247)."),
            ),
        ] = None,
    ) -> None:
        """Run the FastAPI server (and the optional mitmproxy ingress)."""
        bootstrap_and_serve(
            host=host,
            port=port,
            enable_mitm=enable_mitm,
            mitm_host=mitm_host,
            mitm_port=mitm_port,
        )
