"""Process orchestrator: run FastAPI and (optional) mitmproxy on one loop.

When ``ingress.mitm.enabled`` is true in ``magos.yaml``, an embedded
``DumpMaster`` runs alongside uvicorn as a sibling asyncio task. A
client pointing ``HTTPS_PROXY`` at the ingress listener can then reach
magos's routing rules transparently; see ``docs/ingress.md`` for
operator setup.

Bind-address layering for FastAPI:

1. ``--host`` / ``--port`` CLI flags (poked into env by ``__main__``)
2. ``MAGOS_HOST`` / ``MAGOS_PORT`` env vars
3. ``ingress.http.host`` / ``ingress.http.port`` in ``magos.yaml``
4. Schema defaults (127.0.0.1 / 6246) in :class:`HttpIngressConfig`

Steps 1+2 funnel through :class:`MagosSettings`; this module merges
that with the yaml block via :func:`resolve_bind`.
"""

from __future__ import annotations

import asyncio

import uvicorn

from magos import __version__
from magos.config.loader import MagosConfig, load_full_config
from magos.config.schema import HttpIngressConfig, MitmIngressConfig
from magos.config.settings import MagosSettings
from magos.ingress.http import create_app
from magos.ingress.mitm.log_bridge import install_log_bridge
from magos.ingress.mitm.master import build_ingress_master
from magos.telemetry import get_logger

log = get_logger("magos.serve")

# How long to wait for FastAPI to finish lifespan startup before kicking
# off the mitm task. Polling is fine here: uvicorn flips ``started``
# only after the lifespan completes (Headroom warmup, registry refresher
# init), so the mitm listener can't accept a request that races a
# half-warm app.
_FASTAPI_READY_POLL_SECONDS = 0.05


def resolve_bind(settings: MagosSettings, http_cfg: HttpIngressConfig) -> tuple[str, int]:
    """Resolve FastAPI bind host/port from env-or-yaml, env winning.

    Treats empty strings from env as unset so an accidentally-blank
    ``MAGOS_HOST=`` doesn't shadow a real yaml default.
    """
    host = settings.host or http_cfg.host
    port = settings.port if settings.port is not None else http_cfg.port
    return host, port


def resolve_mitm(settings: MagosSettings, mitm_cfg: MitmIngressConfig) -> MitmIngressConfig:
    """Merge ``MAGOS_MITM_*`` env overrides over the yaml ``ingress.mitm`` block.

    Each env var is optional; an unset value leaves the yaml field
    untouched. ``MAGOS_MITM_HOST=""`` is treated as unset (mirroring
    :func:`resolve_bind`) so an accidentally-blank export doesn't shadow
    a real yaml default.
    """
    enabled = settings.mitm_enabled if settings.mitm_enabled is not None else mitm_cfg.enabled
    host = settings.mitm_host or mitm_cfg.host
    port = settings.mitm_port if settings.mitm_port is not None else mitm_cfg.port
    intercept_hosts = (
        settings.mitm_intercept_hosts
        if settings.mitm_intercept_hosts is not None
        else mitm_cfg.intercept_hosts
    )
    return MitmIngressConfig(
        enabled=enabled,
        host=host,
        port=port,
        intercept_hosts=intercept_hosts,
    )


def serve(*, settings: MagosSettings | None = None) -> None:
    """Synchronous entrypoint: build app, run orchestrator under asyncio."""
    settings = settings or MagosSettings()
    asyncio.run(serve_async(settings=settings))


async def serve_async(*, settings: MagosSettings) -> None:
    """Run uvicorn + (optional) mitm ingress until either shuts down.

    On first task done (clean exit, crash, or signal), cancels the
    other so a single SIGINT brings the whole process down without
    orphan listeners.
    """
    cfg: MagosConfig = load_full_config(settings.config_path)
    bind_host, bind_port = resolve_bind(settings, cfg.ingress.http)
    mitm_cfg = resolve_mitm(settings, cfg.ingress.mitm)

    log.info(
        "serve.starting",
        version=__version__,
        host=bind_host,
        port=bind_port,
        ingress_enabled=mitm_cfg.enabled,
        config_path=settings.config_path,
    )

    app = create_app(routing=cfg.routing, registry=cfg.registry)
    uvi_config = uvicorn.Config(
        app,
        host=bind_host,
        port=bind_port,
        log_config=None,
        access_log=settings.access_log,
    )
    uvi_server = uvicorn.Server(uvi_config)

    fastapi_task = asyncio.create_task(uvi_server.serve(), name="magos.fastapi")

    if not mitm_cfg.enabled or not mitm_cfg.intercept_hosts:
        if mitm_cfg.enabled:
            log.warning(
                "ingress.no_intercept_hosts",
                hint="ingress.mitm.enabled is true but intercept_hosts is empty; ingress proxy not started",
            )
        await fastapi_task
        return

    # Wait for the FastAPI lifespan to complete (Headroom warmup,
    # registry refresher start, /metrics provider configured) before
    # the ingress listener can accept its first request. uvicorn flips
    # ``Server.started`` when its lifespan event finishes; there is no
    # event we can await directly, so polling is the documented idiom.
    while not uvi_server.started and not fastapi_task.done():  # noqa: ASYNC110
        await asyncio.sleep(_FASTAPI_READY_POLL_SECONDS)
    if fastapi_task.done():
        # FastAPI failed during startup; surface the exception.
        fastapi_task.result()
        return

    install_log_bridge()
    master = build_ingress_master(mitm_cfg, target_host=bind_host, target_port=bind_port)
    mitm_task = asyncio.create_task(master.run(), name="magos.mitm")
    log.info(
        "ingress.started",
        listen=f"{mitm_cfg.host}:{mitm_cfg.port}",
        target=f"{bind_host}:{bind_port}",
        intercept_hosts=list(mitm_cfg.intercept_hosts),
    )

    done, pending = await asyncio.wait(
        {fastapi_task, mitm_task}, return_when=asyncio.FIRST_COMPLETED
    )
    # Whichever finished first triggers shutdown of the other. Use the
    # task's own shutdown hook when available so listeners close cleanly
    # before we cancel.
    if mitm_task in pending:
        master.shutdown()  # type: ignore[no-untyped-call]
    if fastapi_task in pending:
        uvi_server.should_exit = True
    for task in pending:
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except (TimeoutError, asyncio.CancelledError):
            task.cancel()
    # Re-raise the first exception, if any, so process exit code reflects it.
    for task in done:
        task.result()
