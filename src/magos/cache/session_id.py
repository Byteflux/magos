"""Session-id derivation for the ``magos.cache`` tracker store.

Honors ``x-magos-session-id`` if the client supplies one (prefixed
``explicit:``); otherwise hashes (provider + auth-prefix + model +
system-bytes) into a ``derived:<sha1>`` id.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from magos.compression import ProviderName
from magos.shapes import SHAPES, Shape

_AUTH_PREFIX_LEN = 16
_BEARER_PREFIX = "Bearer "

# Compression providers map to a representative wire shape for body
# extraction. Both OpenAI shapes encode system identically (no top-level
# field; first ``role=system`` entry inside ``messages``), so ``openai-chat``
# stands in for both — Responses bodies have neither ``messages`` nor a
# system-prompt field, so the extractor returns empty bytes either way.
_PROVIDER_SHAPE: dict[ProviderName, Shape] = {
    "anthropic": "anthropic",
    "openai": "openai-chat",
}


def derive_session_id(
    headers: Mapping[str, str],
    body: Mapping[str, Any],
    provider: ProviderName,
) -> str:
    """Return a stable session id used to look up the tracker for ``body``."""
    explicit = headers.get("x-magos-session-id", "").strip()
    if explicit:
        return f"explicit:{explicit}"

    auth_prefix = _extract_auth_prefix(headers)
    model = str(body.get("model", "")) or "unknown"
    system_bytes = _extract_system_bytes(body, provider)

    head = f"{provider}|{auth_prefix}|{model}|".encode()
    digest = hashlib.sha1(head + system_bytes, usedforsecurity=False).hexdigest()
    return f"derived:{digest}"


def _extract_auth_prefix(headers: Mapping[str, str]) -> str:
    """First ``_AUTH_PREFIX_LEN`` chars of the api key found in headers, or ``""``."""
    auth = headers.get("authorization", "")
    if auth.startswith(_BEARER_PREFIX):
        return auth[len(_BEARER_PREFIX) : len(_BEARER_PREFIX) + _AUTH_PREFIX_LEN]
    api_key = headers.get("x-api-key", "")
    if api_key:
        return api_key[:_AUTH_PREFIX_LEN]
    return ""


def _extract_system_bytes(body: Mapping[str, Any], provider: ProviderName) -> bytes:
    """Extract system-prompt bytes via the shape's body-field declaration."""
    spec = SHAPES[_PROVIDER_SHAPE[provider]]
    if spec.system_field is not None:
        return _from_top_level_field(body, spec.system_field)
    if spec.messages_field is not None:
        return _from_messages_field(body, spec.messages_field)
    return b""


def _from_top_level_field(body: Mapping[str, Any], field: str) -> bytes:
    """Read a top-level system field that may be a string or a list of text blocks."""
    value = body.get(field, "")
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, list):
        parts: list[str] = [
            block["text"]
            for block in value
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
        return "".join(parts).encode("utf-8")
    return b""


def _from_messages_field(body: Mapping[str, Any], field: str) -> bytes:
    """Read the first ``role=system`` entry from a messages-style list."""
    messages = body.get(field, [])
    if not isinstance(messages, list):
        return b""
    for msg in messages:
        if not (isinstance(msg, dict) and msg.get("role") == "system"):
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content.encode("utf-8")
        if isinstance(content, list):
            parts: list[str] = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and isinstance(b.get("text"), str)
            ]
            return "".join(parts).encode("utf-8")
        return b""
    return b""
