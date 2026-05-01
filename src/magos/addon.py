"""mitmproxy addon: routes Anthropic /v1/messages traffic through the proxy pipeline.

Run with::

    mitmdump -s src/magos/addon.py

This is intentionally thin. All translation and dispatch logic lives in
``magos.proxy``; this module only adapts the mitmproxy flow API.
"""

from __future__ import annotations

import json

from mitmproxy import http

from magos.obs import get_logger
from magos.proxy import proxy_anthropic_messages

log = get_logger("magos.addon")

_ANTHROPIC_MESSAGES_PATH = "/v1/messages"


def _error_response(status: int, message: str) -> http.Response:
    body = json.dumps({"type": "error", "error": {"type": "proxy_error", "message": message}})
    return http.Response.make(status, body, {"content-type": "application/json"})


class MagosAddon:
    async def request(self, flow: http.HTTPFlow) -> None:
        if flow.request.method != "POST" or flow.request.path != _ANTHROPIC_MESSAGES_PATH:
            return

        raw = flow.request.get_text() or ""
        try:
            anthropic_request = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("invalid_json_body", path=flow.request.path)
            flow.response = _error_response(400, "invalid JSON body")
            return

        try:
            anthropic_response = await proxy_anthropic_messages(anthropic_request)
        except Exception as exc:
            log.error("upstream_failure", error=str(exc), error_type=type(exc).__name__)
            flow.response = _error_response(502, f"upstream failure: {exc}")
            return

        flow.response = http.Response.make(
            200,
            json.dumps(anthropic_response),
            {"content-type": "application/json"},
        )


addons = [MagosAddon()]
