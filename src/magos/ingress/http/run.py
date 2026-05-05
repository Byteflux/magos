"""Seam between FastAPI ``Request`` and the routing/egress pipeline.
:func:`run_endpoint` is the single call site for endpoint handlers.
See ``docs/architecture/request-flow.md`` for the full lifecycle and
exception ladder."""

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
    # Offload to a worker thread: routing is sync but can block on the
    # Kompress thread-locked singleton during a cold HF download (5-10s),
    # which would stall the asyncio loop and the embedded mitm TLS stream.
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
