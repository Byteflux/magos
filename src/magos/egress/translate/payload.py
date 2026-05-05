"""LiteLLM payload builder + process-wide behavior toggles.

Per-endpoint shape massaging lives in the sibling endpoint modules. See
``docs/architecture/translation.md``.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, Protocol

import litellm

# Drop unsupported params at the destination instead of raising
# ``UnsupportedParamsError``. Supported providers still receive them (e.g.
# Anthropic ``context_management``), so cross-shape routing tolerates new
# client-side fields without rule-level rewrites.
litellm.drop_params = True

# Body-derived headers the upstream HTTP client generates itself. Forwarding
# the inbound values into ``extra_headers`` overrides the SDK's own and
# breaks body parsing upstream.
_DISPATCH_BLOCKED_HEADERS: frozenset[str] = frozenset(
    {"content-type", "content-length", "content-encoding", "accept-encoding"}
)

# Inbound auth headers; suppressed only when the operator picked an
# upstream key via ``api_key``. The openai-sdk lets ``extra_headers``
# override the ``api_key`` kwarg, so leaking the inbound bearer to a
# different upstream produces a misleading "Invalid API key" 401.
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

    ``dispatch_model`` overrides ``request["model"]``; ``forward_headers``
    feed ``extra_headers`` (auth/version/beta preserved); ``api_key`` /
    ``api_base`` override LiteLLM's per-provider defaults. ``api_base`` is
    required for ``custom_openai`` upstreams (e.g. Vultr), where LiteLLM has
    no built-in host.
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
