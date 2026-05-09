"""`RequestService`: Application Service Layer for the routing pipeline.

Owns the request lifecycle: route, dispatch, return a transport-agnostic
response. Constructed once at app startup; one instance shared by all
requests across both ingress surfaces (FastAPI + mitmproxy).

Raises `pydantic.ValidationError` for LiteLLM translation errors
(wire-shape problem; the HTTP adapter re-raises as 400). All other
failures are folded into the returned `RoutedResponse`.

See `docs/architecture/request-flow.md` for the full lifecycle and
exception ladder.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError
from structlog.contextvars import bind_contextvars

from magos.dispatch import CompletionFn
from magos.dispatch.errors import DispatchError
from magos.dispatch.gateway import Gateway
from magos.routing import (
    Endpoint,
    RoutedRequest,
    RouteError,
    error_envelope,
    format_dispatch_error_message,
)
from magos.routing.engine import Router
from magos.telemetry import get_logger

log = get_logger("magos.service")


@dataclass(frozen=True)
class RoutedResponse:
    """Transport-agnostic response from the routing/egress pipeline."""

    status: int
    headers: dict[str, str]
    # body is set for non-streaming responses; stream is set for streaming.
    body: bytes | dict[str, Any] | None
    stream: AsyncIterator[bytes] | None
    media_type: str | None = None


class RequestService:
    """Application Service: route + dispatch a `RoutedRequest`.

    `completion` is per-endpoint (different LiteLLM SDK calls per
    endpoint), so it's a method parameter rather than constructor
    injection. Phase C2 of the architecture migration replaces this
    with a `Gateway` abstraction selected by `RouteDecision.target.gateway`.
    """

    def __init__(
        self,
        *,
        router: Router,
        gateway: Gateway,
    ) -> None:
        self._router = router
        self._gateway = gateway

    async def process(
        self,
        routed: RoutedRequest,
        completion: CompletionFn,
    ) -> RoutedResponse:
        """Route and dispatch `routed`, returning a transport-agnostic response."""
        # Offload to a worker thread: routing is sync but can block on the
        # Kompress thread-locked singleton during a cold HF download (5-10s),
        # which would stall the asyncio loop and the embedded mitm TLS stream.
        decision_or_err = await asyncio.to_thread(self._router.route, routed)

        if isinstance(decision_or_err, RouteError):
            _log_route_error(decision_or_err)
            return _render_route_error(decision_or_err)

        endpoint: Endpoint = routed.endpoint
        model = str(routed.body.get("model", ""))
        rule = decision_or_err.rule_label()
        gateway = decision_or_err.target.gateway

        # Bind once for the rest of the request; downstream events (dispatch,
        # compress.applied, egress.usage, mid-stream errors) inherit these
        # via structlog.contextvars, so individual log calls can stay terse
        # without losing the routing context. Not unbound: streaming responses
        # are iterated by FastAPI after this method returns; the request task
        # owns its own contextvars and is GC'd when the request ends.
        bind_contextvars(rule=rule, gateway=gateway, endpoint=endpoint, model=model)

        log.debug("route.matched")

        try:
            raw = await self._gateway.dispatch(decision_or_err, completion=completion)
        except DispatchError as exc:
            log.warning("route.dispatch_error", error=str(exc))
            err = RouteError(
                status=503,
                code="dispatch_error",
                message=format_dispatch_error_message(str(exc)),
                model=model,
                endpoint=endpoint,
            )
            return _render_route_error(err)
        except ValidationError:
            # LiteLLM translation-layer error; re-raise so the HTTP adapter can
            # map it to a 400 with the structured Pydantic error detail.
            raise
        except Exception as exc:
            log.error("upstream_failure", error=str(exc), error_type=type(exc).__name__)
            return RoutedResponse(
                status=502,
                headers={},
                body={"detail": f"upstream failure: {exc}"},
                stream=None,
                media_type=None,
            )

        return _wrap_dispatch_result(raw)


# ---------------------------------------------------------------------------
# Internal helpers (module-private; not part of the public Service surface)
# ---------------------------------------------------------------------------


def _log_route_error(err: RouteError) -> None:
    """Emit a structured event for a `RouteError` returned by the router.

    Kept separate from `_render_route_error` so the dispatch-error branch
    (which already logs at WARN at the catch site) can render without
    re-emitting an INFO duplicate.
    """
    event = "route.unmatched" if err.code == "unmatched" else "route.dispatch_error"
    log.info(event, endpoint=err.endpoint, model=err.model, message=err.message)


def _render_route_error(err: RouteError) -> RoutedResponse:
    """Convert a `RouteError` into a transport-agnostic JSON response."""
    body = error_envelope(endpoint=err.endpoint, code=err.code, message=err.message)
    return RoutedResponse(
        status=err.status,
        headers={"content-type": "application/json"},
        body=body,
        stream=None,
        media_type="application/json",
    )


def _wrap_dispatch_result(raw: Any) -> RoutedResponse:
    """Wrap the heterogeneous return of `Gateway.dispatch` into a `RoutedResponse`.

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
