"""``PassthroughGateway``: byte-exact same-shape forwarding."""

from __future__ import annotations

import json
from typing import Any

from fastapi import Response
from fastapi.responses import StreamingResponse

from magos.egress import CompletionFn
from magos.egress.auth import maybe_inject_api_key
from magos.egress.errors import DispatchError
from magos.egress.passthrough import (
    _HTTP_ERROR_THRESHOLD,
    call_passthrough,
    stream_passthrough,
)
from magos.egress.usage import log_usage_from_body, tap_stream
from magos.routing import RouteDecision
from magos.shapes import shape_for_endpoint

from .base import Gateway, make_on_complete


class PassthroughGateway(Gateway):
    """Forward the request bytes verbatim to ``target.base_url + path``."""

    async def dispatch(
        self,
        decision: RouteDecision,
        *,
        completion: CompletionFn,
    ) -> Response | StreamingResponse | dict[str, Any]:
        req = decision.request
        target = decision.target

        if not target.base_url:  # validated at config load; defensive guard.
            raise DispatchError("passthrough rule has no base_url")

        forward_headers = maybe_inject_api_key(dict(req.headers), target)
        is_streaming = bool(req.body.get("stream"))
        on_complete = make_on_complete(req.post_response_hooks)
        body_bytes = req.raw_body if not req.body_dirty else json.dumps(dict(req.body)).encode()
        model_hint = str(req.body.get("model", ""))
        shape = shape_for_endpoint(req.endpoint)

        if is_streaming:
            upstream = stream_passthrough(
                body_bytes,
                forward_headers,
                target.base_url,
                path=req.forward_path,
                method=req.method,
                model_hint=model_hint,
            )
            iterator = (
                tap_stream(
                    upstream,
                    shape,
                    endpoint=req.endpoint,
                    fallback_model=model_hint or None,
                    on_complete=on_complete,
                )
                if shape is not None
                else upstream
            )
            return StreamingResponse(iterator, media_type="text/event-stream")

        status, raw, content_type = await call_passthrough(
            body_bytes,
            forward_headers,
            target.base_url,
            path=req.forward_path,
            method=req.method,
            model_hint=model_hint,
        )
        if (
            shape is not None
            and status < _HTTP_ERROR_THRESHOLD
            and content_type.startswith("application/json")
        ):
            try:
                parsed = json.loads(raw)
            except (UnicodeDecodeError, ValueError):
                parsed = None
            if parsed is not None:
                log_usage_from_body(shape, parsed, endpoint=req.endpoint, on_complete=on_complete)
        return Response(content=raw, status_code=status, media_type=content_type)
