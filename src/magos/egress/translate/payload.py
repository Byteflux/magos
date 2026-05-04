"""Shared LiteLLM payload builder + cross-shape behavior toggles.

Process-wide LiteLLM tweaks live here so they're applied once on import.
The payload builder is reused by every translate path; per-endpoint
shape massaging (Anthropic extras stripping, etc.) lives in the
sibling endpoint modules.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, Protocol

import litellm

# Cross-shape translation routinely surfaces params one provider supports and
# another does not (e.g. Anthropic's ``context_management`` arriving on a
# request routed to ``custom_openai`` for Vultr). LiteLLM's per-provider
# allow-lists raise ``UnsupportedParamsError`` by default; flipping
# ``drop_params`` to True makes it silently drop unsupported params at the
# destination only — supported providers (Anthropic for ``context_management``)
# still receive them. Without this, every new client-side feature that lands
# in Claude Code or the OpenAI SDK breaks routing to alt providers until we
# patch a request rewrite.
litellm.drop_params = True

# Headers that the upstream HTTP client (litellm/openai-sdk/httpx) generates
# from the serialized request body. Forwarding the inbound values into
# ``extra_headers`` conflicts with that machinery: e.g. an inbound
# ``content-type: application/json`` overrides the SDK's own header and the
# upstream sees a body it cannot parse, returning "you must provide a model
# parameter". Server-level blocking (ingress.http.headers) is for the
# byte-exact passthrough path which legitimately needs ``content-type``.
_DISPATCH_BLOCKED_HEADERS: frozenset[str] = frozenset(
    {"content-type", "content-length", "content-encoding", "accept-encoding"}
)

# Auth headers describing the inbound (client -> magos) hop. When the
# operator has chosen an upstream key explicitly via ``api_key``, these
# must NOT be forwarded into ``extra_headers``: the openai-sdk lets
# ``extra_headers`` override the ``api_key`` kwarg, so leaking the
# inbound bearer to a different upstream provider produces a misleading
# "Invalid API key" 401 even though magos was invoked with a valid key.
# When ``api_key`` is None (rule has no ``api_key_env``), these stay in
# place so litellm's per-provider env-var resolution still wins.
_INBOUND_AUTH_HEADERS: frozenset[str] = frozenset({"authorization", "x-api-key"})


class CompletionFn(Protocol):
    def __call__(self, **kwargs: Any) -> Awaitable[Any]: ...


def resolve_client_model(request_model: str, provider: str | None, dispatch_model: str) -> str:
    """Compute the client-facing model id to write into a translated response.

    LiteLLM yields the dispatch model (e.g. ``custom_openai/Qwen/...``); the
    client expects the namespaced id it sent (or one constructed from the
    routing provider when the request used a bare alias). Falls back to the
    dispatch model only when the request carried no model at all.
    """
    if not request_model:
        return dispatch_model
    if provider and not request_model.startswith(f"{provider}/"):
        return f"{provider}/{request_model}"
    return request_model


def coerce_to_dict(resp: Any) -> dict[str, Any]:
    if hasattr(resp, "model_dump"):
        dumped: dict[str, Any] = resp.model_dump()
        return dumped
    if isinstance(resp, dict):
        return dict(resp)
    raise TypeError(f"completion returned unsupported type: {type(resp).__name__}")


def build_payload(
    request: dict[str, Any],
    *,
    dispatch_model: str,
    forward_headers: dict[str, str] | None,
    api_key: str | None,
    api_base: str | None = None,
) -> dict[str, Any]:
    """Compose the kwargs handed to a LiteLLM SDK call.

    ``dispatch_model`` overrides ``request["model"]`` because the routing
    layer has already chosen the LiteLLM-prefixed identifier; the inbound
    body's model may be a bare alias the operator declared.
    ``forward_headers`` are merged into ``extra_headers`` so upstream sees
    client auth, version pins, and beta flags verbatim, preserving the
    provider's billing shape. ``api_key`` is forwarded to LiteLLM when set
    so a rule's ``api_key_env`` can route across multiple keys per provider.
    ``api_base`` overrides LiteLLM's per-provider default URL; required for
    openai-compatible third parties (e.g. Vultr) routed through the generic
    ``custom_openai`` provider, where LiteLLM has no built-in host to fall
    back on.
    """
    out = dict(request)
    out["model"] = dispatch_model
    if forward_headers:
        blocked = _DISPATCH_BLOCKED_HEADERS
        if api_key is not None:
            # Operator picked the upstream key; don't let the inbound auth
            # header (claude code's anthropic token, etc.) leak into
            # extra_headers and override it on the openai-sdk hop.
            blocked = blocked | _INBOUND_AUTH_HEADERS
        safe = {k: v for k, v in forward_headers.items() if k.lower() not in blocked}
        if safe:
            existing = out.get("extra_headers") or {}
            out["extra_headers"] = {**existing, **safe}
    if api_key is not None:
        out["api_key"] = api_key
    if api_base is not None:
        out["api_base"] = api_base
    return out
