"""Seam between FastAPI's ``Request`` and the routing/egress pipeline.

:func:`run_endpoint` is the single entry point endpoint handlers call.
Steps:

1. Read body bytes; parse JSON to dict (400 on parse error or non-object).
2. Filter inbound headers via
   :func:`magos.ingress.http.headers.forwardable_headers`.
3. Build :class:`magos.routing.RoutedRequest` (frozen dataclass; carries
   ``raw_body``, parsed ``body``, ``method``, ``actual_path``, etc.).
4. Call :func:`magos.routing.route` with the registry config available
   on ``app.state``.
5. On :class:`RouteError` → render the per-endpoint error envelope.
6. On :class:`RouteDecision` → ``await dispatch_decision(...)`` from
   :mod:`magos.egress.dispatch`.

Exception ladder around the dispatch call:

- :class:`DispatchError` → 503 envelope (rule-time failures, e.g.
  missing env var for ``api_key_env``)
- :class:`pydantic.ValidationError` → 400 (translation-layer schema
  rejection from inside LiteLLM)
- :class:`HTTPException` re-raised as-is
- everything else → 502 ``upstream_failure``
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any, cast

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from magos.egress.dispatch import DispatchError, dispatch_decision
from magos.ingress.http.headers import forwardable_headers
from magos.registry.refresher import Refresher
from magos.registry.schema import RegistryYaml
from magos.routing import (
    Endpoint,
    RoutedRequest,
    RouteError,
    RoutingConfig,
    error_envelope,
    format_dispatch_error_message,
    route,
)
from magos.telemetry import get_logger

log = get_logger("magos.ingress.http.run")

CompletionFn = Callable[..., Awaitable[Any]]


async def run_endpoint(
    endpoint: Endpoint,
    request: Request,
    completion: CompletionFn,
    *,
    method: str = "POST",
    actual_path: str | None = None,
) -> Response | StreamingResponse | dict[str, Any]:
    raw_body = await request.body()
    try:
        body: dict[str, Any] = json.loads(raw_body) if raw_body else {}
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="request body must be a JSON object")

    forward = forwardable_headers(request.headers)
    routed = RoutedRequest(
        endpoint=endpoint,
        headers=forward,
        body=body,
        raw_body=raw_body,
        method=cast(Any, method),
        actual_path=actual_path,
    )
    cfg = cast(RoutingConfig, request.app.state.routing)
    refresher = cast("Refresher | None", request.app.state.refresher)
    registry_cfg = cast(RegistryYaml, request.app.state.registry_config)
    # Routing is sync by design but can block on Headroom's Kompress
    # thread-locked singleton during a cold-cache download (the lock is
    # held for the full HF download, easily 5-10s on first run).
    # Offload to a worker thread so the asyncio loop keeps servicing
    # other requests — and, critically, so the embedded mitm proxy can
    # flush bytes back to the client instead of stalling the TLS stream.
    decision_or_err = await asyncio.to_thread(
        route,
        routed,
        cfg,
        registry=refresher.state if refresher is not None else None,
        registry_settings=registry_cfg.registry if refresher is not None else None,
        providers=registry_cfg.providers if refresher is not None else None,
    )

    if isinstance(decision_or_err, RouteError):
        return _render_route_error(decision_or_err)

    log.info(
        "route.matched",
        rule=decision_or_err.rule_label(),
        endpoint=endpoint,
        model=str(routed.body.get("model", "")),
        mode=decision_or_err.action.mode,
    )

    try:
        return await dispatch_decision(decision_or_err, completion=completion)
    except DispatchError as exc:
        log.warning(
            "route.dispatch_error",
            rule=decision_or_err.rule_label(),
            endpoint=endpoint,
            error=str(exc),
        )
        err = RouteError(
            status=503,
            code="dispatch_error",
            message=format_dispatch_error_message(str(exc)),
            model=str(routed.body.get("model", "")),
            endpoint=endpoint,
        )
        return _render_route_error(err)
    except ValidationError as exc:
        # Translation-layer schema check rejected the body; surface as 400.
        raise HTTPException(status_code=400, detail=exc.errors()) from exc
    except HTTPException:
        raise
    except Exception as exc:
        log.error(
            "upstream_failure",
            endpoint=endpoint,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        raise HTTPException(status_code=502, detail=f"upstream failure: {exc}") from exc


def _render_route_error(err: RouteError) -> JSONResponse:
    log.info(
        "route." + ("unmatched" if err.code == "unmatched" else "dispatch_error"),
        endpoint=err.endpoint,
        model=err.model,
        message=err.message,
    )
    body = error_envelope(endpoint=err.endpoint, code=err.code, message=err.message)
    return JSONResponse(status_code=err.status, content=body)
