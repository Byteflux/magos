"""HTTP-level passthrough for same-shape requests.

When a client and the resolved upstream speak the same wire shape, no
translation is needed: forwarding the request bytes and headers verbatim
preserves auth (``Authorization`` bearer or ``x-api-key``), version pins,
beta feature flags, and the provider's billing surface exactly. For
Anthropic specifically it also preserves prompt-cache hashes and avoids
the LiteLLM Anthropic -> OpenAI -> Anthropic re-translation.

The functions here are shape-agnostic: the caller passes ``path``
(``/v1/messages``, ``/v1/responses``, ...) and the dispatcher in
``magos.routing.dispatch`` chooses based on the matched rule's endpoint.
LiteLLM is intentionally NOT involved: it short-circuits on missing
``ANTHROPIC_API_KEY`` before any HTTPS call, which breaks the OAuth
bearer case (Claude Code's auth model).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from magos.obs import get_logger

log = get_logger("magos.passthrough")

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
    model_hint: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncIterator[bytes]:
    """Stream-forward a same-shape request to ``upstream_base_url + path``.

    Forwards raw request bytes verbatim (no JSON parse + re-serialise) so
    the upstream sees a byte-identical body. For Anthropic this is
    required for prompt caching: cache hashes are computed between
    ``cache_control`` breakpoints, and any whitespace shift breaks the
    lookup, billing the request as fresh long-context input.

    Yields raw chunks from the upstream response so the client receives
    the SSE framing exactly as the upstream emitted it.

    ``transport`` is for tests (httpx.MockTransport); production leaves
    it unset so the real network is used.
    """
    url = f"{upstream_base_url.rstrip('/')}{path}"
    log.info("passthrough.stream", url=url, model=model_hint, body_size=len(raw_body))
    async with (
        _make_client(transport) as client,
        client.stream("POST", url, content=raw_body, headers=forward_headers) as resp,
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
    model_hint: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[int, bytes, str]:
    """Non-streaming same-shape passthrough.

    Forwards raw bytes verbatim so prompt caching (Anthropic) is
    preserved. Returns ``(status_code, body_bytes, content_type)`` so
    the server endpoint can mirror the upstream's status and
    content-type back to the client.
    """
    url = f"{upstream_base_url.rstrip('/')}{path}"
    log.info("passthrough.call", url=url, model=model_hint, body_size=len(raw_body))
    async with _make_client(transport) as client:
        resp = await client.post(url, content=raw_body, headers=forward_headers)
    return (
        resp.status_code,
        resp.content,
        resp.headers.get("content-type", "application/json"),
    )


def httpx_text_to_json(raw: bytes) -> str:
    """Tiny helper: best-effort JSON-string-encode an upstream error body."""
    try:
        return json.dumps(raw.decode("utf-8", errors="replace"))
    except Exception:
        return '""'
