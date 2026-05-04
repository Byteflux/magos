"""Route-level error types and per-endpoint error envelopes.

The router can fail in three ways:

- **Unmatched** (404): every rule's match expression rejected the request.
- **Dispatch error** (503): a rule matched, but its post-rewrites failed
  at runtime (e.g., ``jq_patch`` returned a non-object), or downstream
  dispatch hit a config invariant (missing ``api_key_env`` value, unknown
  provider).
- **Upstream failure** (502): the dispatcher reached the upstream and the
  upstream returned an error or the connection failed. This is handled by
  the existing 502 wrapper in ``magos.ingress.http.run`` and is not produced here.

The HTTP body shape mirrors the inbound endpoint: Anthropic error envelope
for ``/v1/messages*``, OpenAI error envelope for ``/v1/chat/completions``.
This keeps clients from seeing one shape on success and a different shape
on routing-layer failure.
"""

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
    """Render the error body in the shape native to ``endpoint``.

    Anthropic and OpenAI use different JSON shapes. ``code`` selects the
    error-type token within each shape so clients see a familiar string.
    """
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
    """Static hint plus the inbound model name for caller debugging."""
    if model:
        return f"no route in magos.yaml matched this request (model={model!r})"
    return "no route in magos.yaml matched this request"


def format_dispatch_error_message(reason: str) -> str:
    """Static prefix plus a short reason; never echoes secrets or env names."""
    return f"route configuration error: {reason}"
