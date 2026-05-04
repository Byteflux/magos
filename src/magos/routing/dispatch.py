"""Bridge from a ``RouteDecision`` to the existing proxy/passthrough seams.

The dispatcher is the only routing-layer module that knows about FastAPI
response types. ``server.py`` calls ``dispatch_decision`` with a decision
already produced by ``route()``; the dispatcher then picks the right
underlying call based on endpoint, ``action.mode``, and the request's
``stream`` flag.

API-key handling:

- ``mode: translate``: ``action.api_key_env`` (when set) is read from the
  process environment and passed as ``api_key=`` to the proxy functions,
  which forward it to litellm. Lets one provider use multiple keys (e.g.
  tier-routing) by declaring separate rules with different env vars.
- ``mode: passthrough``: when neither ``Authorization`` nor ``x-api-key``
  is present in the inbound headers and ``action.api_key_env`` is set, we
  inject the env value into the forwarded headers. The shape is provider-
  aware: ``provider: anthropic`` -> ``x-api-key: <env>`` (the Anthropic
  API convention), every other provider -> ``Authorization: Bearer <env>``
  (the openai-compatible convention used by openai, openrouter, vultr,
  etc.). ``action.auth_header`` overrides the default. Claude-Code-style
  Anthropic OAuth tokens (``sk-ant-oat...``) are detected and sent as
  ``Authorization: Bearer ...`` plus ``anthropic-beta: oauth-2025-04-20``
  regardless of the default/override, since that's the only shape
  api.anthropic.com accepts for that credential class. Headers are not
  part of the prompt-cache hash, so this injection does not break
  byte-exact billing.

Env-var lookup failures surface as ``DispatchError``; ``server.py`` turns
them into the 503 ``dispatch_error`` envelope.
"""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Response
from fastapi.responses import StreamingResponse

from magos.obs import get_logger
from magos.passthrough import call_passthrough, stream_passthrough
from magos.proxy import (
    proxy_anthropic_messages,
    proxy_openai_chat_completions,
    proxy_openai_responses,
    stream_anthropic_messages,
    stream_openai_chat_completions,
    stream_openai_responses,
)
from magos.routing.engine import RouteDecision
from magos.routing.models import Action
from magos.tokens import count_tokens

log = get_logger("magos.routing.dispatch")

CompletionFn = Callable[..., Awaitable[Any]]


class DispatchError(Exception):
    """Raised when a runtime config invariant fails (e.g., missing env var)."""


async def dispatch_decision(  # noqa: PLR0911
    decision: RouteDecision,
    *,
    completion: CompletionFn,
) -> Response | StreamingResponse | dict[str, Any]:
    """Hand ``decision`` off to the right downstream call site.

    Branches: count_tokens, passthrough+stream, passthrough+non-stream,
    translate x {messages, chat, responses} x {stream, non-stream}.
    The ``# noqa`` is a deliberate suppression of the per-function
    return-cap; collapsing branches into helpers makes the dispatch shape
    harder to read, not easier.
    """
    req = decision.request
    action = decision.action

    if req.endpoint == "/v1/messages/count_tokens":
        return await _dispatch_count_tokens(decision, completion=completion)

    forward_headers = _maybe_inject_api_key(dict(req.headers), action)
    is_streaming = bool(req.body.get("stream"))

    if action.mode == "passthrough":
        if not action.base_url:  # validated at config load; defensive guard.
            raise DispatchError("passthrough rule has no base_url")
        body_bytes = req.raw_body if not req.body_dirty else json.dumps(dict(req.body)).encode()
        model_hint = str(req.body.get("model", ""))
        # Templated endpoints (/v1/responses/{id}) keep the raw inbound path
        # for forwarding via ``forward_path``; for fixed endpoints the two
        # are identical.
        if is_streaming:
            return StreamingResponse(
                stream_passthrough(
                    body_bytes,
                    forward_headers,
                    action.base_url,
                    path=req.forward_path,
                    method=req.method,
                    model_hint=model_hint,
                ),
                media_type="text/event-stream",
            )
        status, raw, content_type = await call_passthrough(
            body_bytes,
            forward_headers,
            action.base_url,
            path=req.forward_path,
            method=req.method,
            model_hint=model_hint,
        )
        return Response(content=raw, status_code=status, media_type=content_type)

    # mode: translate -- only POST endpoints have litellm equivalents.
    if req.method != "POST":
        raise DispatchError(
            f"mode='translate' does not support method={req.method!r}; "
            "use mode='passthrough' for auxiliary GET/DELETE endpoints"
        )
    api_key = _resolve_api_key(action.api_key_env)
    api_base = action.base_url

    if req.endpoint == "/v1/messages":
        if is_streaming:
            stream = stream_anthropic_messages(
                dict(req.body),
                dispatch_model=decision.dispatch_model,
                completion=completion,
                forward_headers=forward_headers,
                api_key=api_key,
                api_base=api_base,
            )
            return StreamingResponse(stream, media_type="text/event-stream")
        return await proxy_anthropic_messages(
            dict(req.body),
            dispatch_model=decision.dispatch_model,
            completion=completion,
            forward_headers=forward_headers,
            api_key=api_key,
            api_base=api_base,
        )

    if req.endpoint == "/v1/chat/completions":
        if is_streaming:
            stream = stream_openai_chat_completions(
                dict(req.body),
                dispatch_model=decision.dispatch_model,
                completion=completion,
                forward_headers=forward_headers,
                api_key=api_key,
                api_base=api_base,
            )
            return StreamingResponse(stream, media_type="text/event-stream")
        return await proxy_openai_chat_completions(
            dict(req.body),
            dispatch_model=decision.dispatch_model,
            completion=completion,
            forward_headers=forward_headers,
            api_key=api_key,
            api_base=api_base,
        )

    # /v1/responses
    if is_streaming:
        stream = stream_openai_responses(
            dict(req.body),
            dispatch_model=decision.dispatch_model,
            completion=completion,
            forward_headers=forward_headers,
            api_key=api_key,
            api_base=api_base,
        )
        return StreamingResponse(stream, media_type="text/event-stream")
    return await proxy_openai_responses(
        dict(req.body),
        dispatch_model=decision.dispatch_model,
        completion=completion,
        forward_headers=forward_headers,
        api_key=api_key,
        api_base=api_base,
    )


async def _dispatch_count_tokens(
    decision: RouteDecision, *, completion: CompletionFn
) -> dict[str, int]:
    """Dispatch a count_tokens request via ``litellm.acount_tokens``.

    LiteLLM auto-picks between the local tokenizer and the upstream's
    native count-tokens endpoint based on the model's provider, so a
    single call covers what was historically split across ``count_locally``
    and ``count_tokens_mode: passthrough``.
    """
    body = dict(decision.request.body)
    n = await count_tokens(
        body,
        dispatch_model=decision.dispatch_model,
        count=completion,
    )
    return {"input_tokens": n}


_ANTHROPIC_OAUTH_TOKEN_PREFIX = "sk-ant-oat"  # noqa: S105
_ANTHROPIC_OAUTH_BETA = "oauth-2025-04-20"


def _maybe_inject_api_key(headers: dict[str, str], action: Action) -> dict[str, str]:
    """In passthrough mode, inject the env-resolved API key when absent.

    Shape is provider-aware: anthropic -> ``x-api-key`` (Anthropic's
    official header), everything else -> ``Authorization: Bearer``
    (openai-compatible convention). ``action.auth_header`` overrides
    the default. Claude-Code-style Anthropic OAuth tokens
    (``sk-ant-oat...``) override both: they always go out as a Bearer
    plus the ``anthropic-beta: oauth-2025-04-20`` opt-in header, since
    api.anthropic.com 401s on ``x-api-key`` for that credential class.
    Skipped entirely when the inbound request already carries
    ``Authorization`` or ``x-api-key``.
    """
    if action.mode != "passthrough" or not action.api_key_env:
        return headers
    if "authorization" in headers or "x-api-key" in headers:
        return headers
    value = os.environ.get(action.api_key_env)
    if not value:
        raise DispatchError(f"env var {action.api_key_env!r} is not set")
    if action.provider == "anthropic" and value.startswith(_ANTHROPIC_OAUTH_TOKEN_PREFIX):
        return {
            **headers,
            "authorization": f"Bearer {value}",
            "anthropic-beta": _ANTHROPIC_OAUTH_BETA,
        }
    shape = action.auth_header or _default_auth_header(action.provider)
    if shape == "x-api-key":
        return {**headers, "x-api-key": value}
    return {**headers, "authorization": f"Bearer {value}"}


def _default_auth_header(provider: str) -> str:
    """Pick the auth-header shape for a provider when no override is set."""
    return "x-api-key" if provider == "anthropic" else "bearer"


def _resolve_api_key(api_key_env: str | None) -> str | None:
    """Translate-mode helper: read ``api_key_env`` from the environment."""
    if not api_key_env:
        return None
    value = os.environ.get(api_key_env)
    if not value:
        raise DispatchError(f"env var {api_key_env!r} is not set")
    return value
