"""Transport-agnostic request processing core.

:func:`process_routed_request` takes a :class:`~magos.routing.RoutedRequest`
and returns a :class:`RoutedResponse` without importing any FastAPI types.
The FastAPI handler in :mod:`magos.ingress.http.run` is the thin adapter
that converts ``Request`` -> ``RoutedRequest`` and ``RoutedResponse`` ->
``Response / StreamingResponse``.

See ``docs/architecture/request-flow.md`` for the full lifecycle and
exception ladder.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from magos.egress.dispatch import dispatch_decision
from magos.egress.errors import DispatchError
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

log = get_logger("magos.process")

CompletionFn = Callable[..., Awaitable[Any]]


@dataclass(frozen=True)
class RoutedResponse:
    """Transport-agnostic response from the routing/egress pipeline."""

    status: int
    headers: dict[str, str]
    # body is set for non-streaming responses; stream is set for streaming.
    body: bytes | dict[str, Any] | None
    stream: AsyncIterator[bytes] | None
    media_type: str | None = None


async def process_routed_request(
    routed: RoutedRequest,
    *,
    cfg: RoutingConfig,
    refresher: Refresher | None,
    registry_cfg: RegistryYaml,
    completion: CompletionFn,
) -> RoutedResponse:
    """Route and dispatch ``routed``, returning a transport-agnostic response.

    Raises :class:`pydantic.ValidationError` for LiteLLM translation errors
    (wire-shape problem; the HTTP adapter re-raises as 400). All other
    failures are folded into the returned :class:`RoutedResponse`.
    """
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
        pins=registry_cfg.pins if refresher is not None else None,
        provider_order=registry_cfg.provider_order if refresher is not None else (),
    )

    if isinstance(decision_or_err, RouteError):
        return _render_route_error(decision_or_err)

    endpoint: Endpoint = routed.endpoint
    log.info(
        "route.matched",
        rule=decision_or_err.rule_label(),
        endpoint=endpoint,
        model=str(routed.body.get("model", "")),
        mode=decision_or_err.action.mode,
    )

    try:
        raw = await dispatch_decision(decision_or_err, completion=completion)
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
    except ValidationError:
        # LiteLLM translation-layer error; re-raise so the HTTP adapter can
        # map it to a 400 with the structured Pydantic error detail.
        raise
    except Exception as exc:
        log.error(
            "upstream_failure",
            endpoint=endpoint,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return RoutedResponse(
            status=502,
            headers={},
            body={"detail": f"upstream failure: {exc}"},
            stream=None,
            media_type=None,
        )

    return _wrap_dispatch_result(raw)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _render_route_error(err: RouteError) -> RoutedResponse:
    log.info(
        "route." + ("unmatched" if err.code == "unmatched" else "dispatch_error"),
        endpoint=err.endpoint,
        model=err.model,
        message=err.message,
    )
    body = error_envelope(endpoint=err.endpoint, code=err.code, message=err.message)
    return RoutedResponse(
        status=err.status,
        headers={"content-type": "application/json"},
        body=body,
        stream=None,
        media_type="application/json",
    )


def _wrap_dispatch_result(raw: Any) -> RoutedResponse:
    """Wrap the heterogeneous return of ``dispatch_decision`` into a ``RoutedResponse``.

    Uses duck-typing rather than FastAPI isinstance checks so this module
    stays transport-agnostic (no fastapi/starlette imports).
    """
    if hasattr(raw, "body_iterator"):
        # StreamingResponse: AsyncIterator[bytes] + media_type + status_code.
        media: str = raw.media_type or "text/event-stream"
        return RoutedResponse(
            status=raw.status_code,
            headers={},
            body=None,
            stream=raw.body_iterator,
            media_type=media,
        )
    if hasattr(raw, "body"):
        # Response: bytes body + optional media_type + status_code.
        content: bytes | None = raw.body
        return RoutedResponse(
            status=raw.status_code,
            headers={},
            body=content,
            stream=None,
            media_type=raw.media_type,
        )
    # dict — count_tokens and translate non-streaming return dicts.
    return RoutedResponse(
        status=200,
        headers={},
        body=raw,
        stream=None,
        media_type="application/json",
    )
