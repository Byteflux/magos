"""HTTP-level passthrough for same-shape Anthropic requests.

When a client speaks the Anthropic Messages shape and the resolved upstream
is Anthropic, no translation is needed: forwarding the request body and
headers verbatim preserves auth (Authorization bearer or x-api-key),
``anthropic-version`` pins, ``anthropic-beta`` feature flags, and the
provider's billing surface exactly. It also avoids the round-trip of
Anthropic -> OpenAI -> Anthropic that LiteLLM would otherwise perform.

LiteLLM is intentionally NOT involved here: it short-circuits on missing
``ANTHROPIC_API_KEY`` before any HTTPS call, which breaks the OAuth bearer
case (Claude Code's auth model).
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


async def stream_anthropic_passthrough(
    raw_body: bytes,
    forward_headers: dict[str, str],
    upstream_base_url: str,
    *,
    model_hint: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncIterator[bytes]:
    """Stream-forward an Anthropic /v1/messages request to upstream.

    Forwards the raw request bytes verbatim (no JSON parse + re-serialise) so
    the upstream sees a byte-identical body. Required for prompt caching:
    Anthropic hashes content between ``cache_control`` breakpoints, and any
    whitespace shift breaks the cache lookup, billing the request as fresh
    long-context input.

    Yields raw chunks from the upstream response so the client receives the
    SSE framing exactly as Anthropic emitted it, with no re-encoding.

    ``transport`` is for tests (httpx.MockTransport); production leaves it
    unset so the real network is used.
    """
    url = f"{upstream_base_url.rstrip('/')}/v1/messages"
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


async def call_anthropic_passthrough(
    raw_body: bytes,
    forward_headers: dict[str, str],
    upstream_base_url: str,
    *,
    model_hint: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[int, bytes, str]:
    """Non-streaming Anthropic passthrough.

    Forwards the raw request bytes verbatim so prompt caching is preserved.
    Returns ``(status_code, body_bytes, content_type)`` so the server endpoint
    can mirror the upstream's status and content-type back to the client.
    """
    url = f"{upstream_base_url.rstrip('/')}/v1/messages"
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
