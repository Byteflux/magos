"""``/admin/registry/*`` operator endpoints.

Exposed only when a :class:`Refresher` is active (i.e.
``providers:`` non-empty in ``magos.yaml``). The CLI reads from these
to show server-state and trigger out-of-band refreshes:

- ``GET /admin/registry``                : snapshot of in-memory state
- ``POST /admin/registry/refresh``       : refresh one or all providers
- ``POST /admin/registry/prune``         : sweep deprecation past grace

``magos models list/show`` fall back to disk when the server is down,
so these endpoints are convenience surfaces, not the only way to read
the registry.
"""

from __future__ import annotations

from typing import cast

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from magos.registry.refresher import Refresher
from magos.registry.schema import RegistryYaml


def mount_admin_registry_endpoints(app: FastAPI) -> None:
    """Register the three ``/admin/registry/*`` endpoints on ``app``."""
    from magos.registry.discovery.base import DiscoveryError  # noqa: PLC0415
    from magos.registry.store import serialize  # noqa: PLC0415

    @app.get("/admin/registry", include_in_schema=False)
    async def get_registry(request: Request) -> Response:
        refresher = cast(Refresher, request.app.state.refresher)
        return Response(content=serialize(refresher.state), media_type="application/json")

    @app.post("/admin/registry/refresh", include_in_schema=False)
    async def refresh_registry(request: Request, provider: str | None = None) -> Response:
        refresher = cast(Refresher, request.app.state.refresher)
        registry_cfg = cast(RegistryYaml, request.app.state.registry_config)
        targets = [provider] if provider else list(registry_cfg.providers)
        unknown = [p for p in targets if p not in registry_cfg.providers]
        if unknown:
            raise HTTPException(
                status_code=404, detail=f"unknown provider(s): {', '.join(unknown)}"
            )
        refreshed: list[str] = []
        failed: dict[str, str] = {}
        for name in targets:
            try:
                await refresher.refresh(name)
                refreshed.append(name)
            except DiscoveryError as exc:
                failed[name] = str(exc)
        return JSONResponse({"refreshed": refreshed, "failed": failed})

    @app.post("/admin/registry/prune", include_in_schema=False)
    async def prune_registry(request: Request) -> Response:
        """Trigger a prune by refreshing every provider.

        The deprecation state machine drops past-grace entries on every
        successful refresh, so a full refresh round is the simplest way
        to surface the operator-visible "prune" action without adding
        a separate code path.
        """
        refresher = cast(Refresher, request.app.state.refresher)
        registry_cfg = cast(RegistryYaml, request.app.state.registry_config)
        before = sum(1 for e in refresher.state.entries.values() if e.is_deprecated)
        for name in registry_cfg.providers:
            try:
                await refresher.refresh(name)
            except DiscoveryError:
                continue  # other providers still get their chance
        after = sum(1 for e in refresher.state.entries.values() if e.is_deprecated)
        return JSONResponse({"deprecated_before": before, "deprecated_after": after})
