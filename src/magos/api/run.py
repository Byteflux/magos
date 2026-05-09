"""FastAPI adapter: `Request` -> `RoutedRequest` -> `RequestService.process` -> `Response`.

Body parsing and JSON shape validation happen here (FastAPI-specific errors).
All routing/dispatch logic lives in `magos.service.request`.
See `docs/architecture/request-flow.md` for the full lifecycle and
exception ladder.

`run_endpoint` also binds a `request_id` via `structlog.contextvars` so
every log line emitted inside the request -- including those that fire
mid-stream after the handler returns -- carries the same correlation id.
The id is read from the inbound `X-Request-ID` header when present and
generated otherwise; a 12-char hex token is short enough for line-by-line
greppability while staying unique per request.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, cast

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError
from structlog.contextvars import bind_contextvars

from magos.api.headers import forwardable_headers
from magos.dispatch import CompletionFn
from magos.routing import Endpoint, RoutedRequest
from magos.service import RequestService, RoutedResponse

__all__ = ["CompletionFn", "run_endpoint"]


def _request_id_from(request: Request) -> str:
    """Return the inbound `X-Request-ID` (trimmed, capped) or a fresh hex token."""
    incoming = request.headers.get("x-request-id", "").strip()
    if incoming:
        return incoming[:64]
    return uuid.uuid4().hex[:12]


async def run_endpoint(
    endpoint: Endpoint,
    request: Request,
    completion: CompletionFn,
    *,
    method: str = "POST",
    actual_path: str | None = None,
) -> Response | StreamingResponse | dict[str, Any]:
    """Thin FastAPI adapter around `RequestService.process`.

    Steps:
    1. Read raw body bytes.
    2. Parse JSON; raise `HTTPException(400)` for parse / shape errors.
    3. Build `RoutedRequest`.
    4. Delegate to `app.state.service.process`.
    5. Adapt `RoutedResponse` to a FastAPI `Response`.
    """
    # Bind for the lifetime of the request task. We intentionally do NOT
    # unbind here: streaming responses are iterated by FastAPI *after* the
    # handler returns, and mid-stream events (`egress.usage` from the
    # tap, `egress.stream_error` from `safe_stream`) need the same id.
    # Each FastAPI request runs in its own asyncio task with a fresh
    # contextvars context, so isolation between requests is automatic.
    bind_contextvars(request_id=_request_id_from(request))
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
    service = cast(RequestService, request.app.state.service)

    try:
        result = await service.process(routed, completion)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc

    return _adapt(result)


def _adapt(result: RoutedResponse) -> Response | StreamingResponse | dict[str, Any]:
    """Convert a `RoutedResponse` to a FastAPI response type."""
    if result.stream is not None:
        return StreamingResponse(result.stream, media_type=result.media_type or "text/event-stream")
    if isinstance(result.body, dict):
        return JSONResponse(status_code=result.status, content=result.body)
    if isinstance(result.body, bytes):
        return Response(
            content=result.body,
            status_code=result.status,
            media_type=result.media_type,
        )
    # body is None — unexpected; raise a generic HTTPException as a backstop.
    raise HTTPException(status_code=result.status, detail="upstream failure")
