"""Session-id derivation for the `magos.compression.tracker` tracker store.

Honors `x-magos-session-id` if the client supplies one (prefixed
`explicit:`); otherwise hashes (provider + auth-prefix + model +
system-bytes) into a `derived:<sha1>` id.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from magos.compression import ProviderName
from magos.shapes import ANTHROPIC, OPENAI_CHAT, Shape

_AUTH_PREFIX_LEN = 16
_BEARER_PREFIX = "Bearer "

# Compression providers map to a representative wire shape for body
# extraction. Both OpenAI shapes encode system identically (no top-level
# field; first `role=system` entry inside `messages`), so
# `OPENAI_CHAT` stands in for both — Responses bodies have neither
# `messages` nor a system-prompt field, so the extractor returns empty
# bytes either way.
_PROVIDER_SHAPE: dict[ProviderName, Shape] = {
    "anthropic": ANTHROPIC,
    "openai": OPENAI_CHAT,
}


def derive_session_id(
    headers: Mapping[str, str],
    body: Mapping[str, Any],
    provider: ProviderName,
) -> str:
    """Return a stable session id used to look up the tracker for `body`."""
    explicit = headers.get("x-magos-session-id", "").strip()
    if explicit:
        return f"explicit:{explicit}"

    auth_prefix = _extract_auth_prefix(headers)
    model = str(body.get("model", "")) or "unknown"
    system_bytes = _PROVIDER_SHAPE[provider].extract_system_bytes(body)

    head = f"{provider}|{auth_prefix}|{model}|".encode()
    digest = hashlib.sha1(head + system_bytes, usedforsecurity=False).hexdigest()
    return f"derived:{digest}"


def _extract_auth_prefix(headers: Mapping[str, str]) -> str:
    """First `_AUTH_PREFIX_LEN` chars of the api key found in headers, or `""`."""
    auth = headers.get("authorization", "")
    if auth.startswith(_BEARER_PREFIX):
        return auth[len(_BEARER_PREFIX) : len(_BEARER_PREFIX) + _AUTH_PREFIX_LEN]
    api_key = headers.get("x-api-key", "")
    if api_key:
        return api_key[:_AUTH_PREFIX_LEN]
    return ""
