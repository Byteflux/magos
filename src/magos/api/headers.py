"""Inbound header filter (stage 1 of 3). Drops hop-by-hop (RFC 7230)
and content-shaping headers; forwards the rest verbatim to preserve
auth, version pins, and Anthropic prompt-cache hashes. See
``docs/architecture/headers-and-auth.md``."""

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
    """Return inbound headers minus hop-by-hop / content-shaping. Keys
    lowercased for case-insensitive matcher/rewrite lookups."""
    return {k.lower(): v for k, v in headers.items() if k.lower() not in _BLOCKED_FORWARD_HEADERS}
