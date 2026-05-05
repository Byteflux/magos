"""Byte-exact same-shape forwarding via httpx.

See ``docs/architecture/request-flow.md`` for the byte-exactness contract.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from magos.telemetry import get_logger

log = get_logger("magos.egress.passthrough")

_DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)
_HTTP_ERROR_THRESHOLD = 400


def _make_client(transport: httpx.AsyncBaseTransport | None) -> httpx.AsyncClient:
    if transport is not None:
        return httpx.AsyncClient(transport=transport, timeout=_DEFAULT_TIMEOUT)
    return httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT)


async def stream_passthrough(
    raw_body: bytes,
    forward_headers: dict[str, str],
    upstream_base_url: str,
    *,
    path: str,
    method: str = "POST",
    model_hint: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncIterator[bytes]:
    """Stream-forward a same-shape request to ``upstream_base_url + path``.

    Body bytes are forwarded verbatim; any re-serialise breaks Anthropic
    prompt-cache lookup. ``transport`` is a test seam (httpx.MockTransport).
    """
    url = f"{upstream_base_url.rstrip('/')}{path}"
    log.info(
        "passthrough.stream", url=url, method=method, model=model_hint, body_size=len(raw_body)
    )
    async with (
        _make_client(transport) as client,
        client.stream(method, url, content=raw_body, headers=forward_headers) as resp,
    ):
        if resp.status_code >= _HTTP_ERROR_THRESHOLD:
            preview = (await resp.aread())[:500]
            log.warning(
                "passthrough.stream_upstream_error",
                status=resp.status_code,
                body_preview=preview.decode("utf-8", errors="replace"),
            )
            yield (
                f'event: error\ndata: {{"type":"error","error":{{"type":"upstream_error",'
                f'"status":{resp.status_code},'
                f'"message":{httpx_text_to_json(preview)}}}}}\n\n'.encode()
            )
            return
        async for chunk in resp.aiter_bytes():
            if chunk:
                yield chunk


async def call_passthrough(
    raw_body: bytes,
    forward_headers: dict[str, str],
    upstream_base_url: str,
    *,
    path: str,
    method: str = "POST",
    model_hint: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[int, bytes, str]:
    """Non-streaming same-shape passthrough; returns ``(status, body, content_type)``."""
    url = f"{upstream_base_url.rstrip('/')}{path}"
    log.info("passthrough.call", url=url, method=method, model=model_hint, body_size=len(raw_body))
    async with _make_client(transport) as client:
        resp = await client.request(method, url, content=raw_body, headers=forward_headers)
    return (
        resp.status_code,
        resp.content,
        resp.headers.get("content-type", "application/json"),
    )


def httpx_text_to_json(raw: bytes) -> str:
    """Best-effort JSON-string-encode an upstream error body."""
    try:
        return json.dumps(raw.decode("utf-8", errors="replace"))
    except Exception:
        return '""'
