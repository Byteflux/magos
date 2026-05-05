"""Inbound header filter for the FastAPI ingress path.

Hop-by-hop headers (RFC 7230) plus a handful of content-shaping headers
that the outbound HTTP client must own. Everything else is forwarded so
upstream sees the client's auth, version pins, and beta flags verbatim,
which preserves provider billing shape (and Anthropic prompt-cache hash
stability under passthrough).

This is the **first** of three header-filter stages; see
``docs/architecture.md`` "Header forwarding". The other two live in
:mod:`magos.egress.translate`.
"""

from __future__ import annotations

from starlette.datastructures import Headers

_BLOCKED_FORWARD_HEADERS: frozenset[str] = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
        "content-encoding",
        "accept-encoding",
    }
)


def forwardable_headers(headers: Headers) -> dict[str, str]:
    """Return inbound headers minus hop-by-hop and content-shaping ones.

    Keys are lowercased so routing matchers and rewrites can use case-
    insensitive lookups uniformly.
    """
    return {k.lower(): v for k, v in headers.items() if k.lower() not in _BLOCKED_FORWARD_HEADERS}
