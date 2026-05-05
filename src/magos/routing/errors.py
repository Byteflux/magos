"""Route-level error types and per-endpoint error envelopes. See ``docs/routing/errors.md``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from magos.routing.request import Endpoint

ErrorCode = Literal["unmatched", "dispatch_error"]


@dataclass(frozen=True, slots=True)
class RouteError:
    """Routing-layer failure carrying enough context to render a response."""

    status: int
    code: ErrorCode
    message: str
    model: str
    endpoint: Endpoint


_ANTHROPIC_TYPE = {
    "unmatched": "not_found_error",
    "dispatch_error": "api_error",
}
_OPENAI_TYPE = {
    "unmatched": "invalid_request_error",
    "dispatch_error": "server_error",
}
_OPENAI_CODE = {
    "unmatched": "no_route_matched",
    "dispatch_error": "dispatch_error",
}


def error_envelope(*, endpoint: Endpoint, code: ErrorCode, message: str) -> dict[str, Any]:
    """Render the error body in the shape native to ``endpoint``."""
    if endpoint in {
        "/v1/chat/completions",
        "/v1/responses",
        "/v1/responses/{id}",
        "/v1/responses/{id}/input_items",
    }:
        return {
            "error": {
                "message": message,
                "type": _OPENAI_TYPE[code],
                "code": _OPENAI_CODE[code],
            }
        }
    return {
        "type": "error",
        "error": {"type": _ANTHROPIC_TYPE[code], "message": message},
    }


def format_unmatched_message(model: str) -> str:
    if model:
        return f"no route in magos.yaml matched this request (model={model!r})"
    return "no route in magos.yaml matched this request"


def format_dispatch_error_message(reason: str) -> str:
    """Static prefix plus reason; never echoes secrets or env names."""
    return f"route configuration error: {reason}"
