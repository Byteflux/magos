"""Bridge from a ``RouteDecision`` to the existing proxy/passthrough seams.

The dispatcher is the only routing-layer module that knows about FastAPI
response types. ``server.py`` calls ``dispatch_decision`` with a decision
already produced by ``route()``; the dispatcher then picks the right
underlying call based on endpoint, ``action.mode``,
``action.count_tokens_mode``, and the request's ``stream`` flag.

API-key handling:

- ``mode: translate``: ``action.api_key_env`` (when set) is read from the
  process environment and passed as ``api_key=`` to the proxy functions,
  which forward it to litellm. Lets one provider use multiple keys (e.g.
  tier-routing) by declaring separate rules with different env vars.
- ``mode: passthrough``: when neither ``Authorization`` nor ``x-api-key``
  is present in the inbound headers and ``action.api_key_env`` is set, we
  inject ``x-api-key: <env>`` into the forwarded headers. Headers are not
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
from magos.tokens import PASSTHROUGH_DISPATCH, count_locally

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

    Six branches: count_tokens, passthrough+stream, passthrough+non-stream,
    translate+messages+stream, translate+messages+non-stream, translate+
    chat+stream, translate+chat+non-stream. The ``# noqa`` is a deliberate
    suppression of the per-function return-cap; collapsing branches into
    helpers makes the dispatch shape harder to read, not easier.
    """
    req = decision.request
    action = decision.action

    if req.endpoint == "/v1/messages/count_tokens":
        return await _dispatch_count_tokens(decision)

    forward_headers = _maybe_inject_api_key(dict(req.headers), action.api_key_env, action.mode)
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

    if req.endpoint == "/v1/messages":
        if is_streaming:
            stream = stream_anthropic_messages(
                dict(req.body),
                dispatch_model=decision.dispatch_model,
                completion=completion,
                forward_headers=forward_headers,
                api_key=api_key,
            )
            return StreamingResponse(stream, media_type="text/event-stream")
        return await proxy_anthropic_messages(
            dict(req.body),
            dispatch_model=decision.dispatch_model,
            completion=completion,
            forward_headers=forward_headers,
            api_key=api_key,
        )

    if req.endpoint == "/v1/chat/completions":
        if is_streaming:
            stream = stream_openai_chat_completions(
                dict(req.body),
                dispatch_model=decision.dispatch_model,
                completion=completion,
                forward_headers=forward_headers,
                api_key=api_key,
            )
            return StreamingResponse(stream, media_type="text/event-stream")
        return await proxy_openai_chat_completions(
            dict(req.body),
            dispatch_model=decision.dispatch_model,
            completion=completion,
            forward_headers=forward_headers,
            api_key=api_key,
        )

    # /v1/responses
    if is_streaming:
        stream = stream_openai_responses(
            dict(req.body),
            dispatch_model=decision.dispatch_model,
            completion=completion,
            forward_headers=forward_headers,
            api_key=api_key,
        )
        return StreamingResponse(stream, media_type="text/event-stream")
    return await proxy_openai_responses(
        dict(req.body),
        dispatch_model=decision.dispatch_model,
        completion=completion,
        forward_headers=forward_headers,
        api_key=api_key,
    )


async def _dispatch_count_tokens(decision: RouteDecision) -> dict[str, int]:
    """Dispatch a count_tokens request via the strategy named by ``action``.

    Auth is intentionally not injected here: the registered passthrough impls
    (currently ``_anthropic_passthrough``) use their own SDK which reads
    ``ANTHROPIC_API_KEY`` from the process environment. Injecting an
    ``x-api-key`` header would either duplicate or override the SDK's own
    auth and the upstream would reject it as ``invalid x-api-key``. Inbound
    headers are still forwarded so beta flags and version pins reach the
    SDK via ``extra_headers`` (with auth headers filtered inside the impl).
    """
    req = decision.request
    action = decision.action
    body = dict(req.body)
    if action.count_tokens_mode == "passthrough":
        impl = PASSTHROUGH_DISPATCH.get(action.provider)
        if impl is None:
            raise DispatchError(
                f"count_tokens_mode='passthrough' with no implementation for "
                f"provider={action.provider!r}"
            )
        return {"input_tokens": await impl(body, forward_headers=dict(req.headers))}
    return {"input_tokens": count_locally(body)}


def _maybe_inject_api_key(
    headers: dict[str, str], api_key_env: str | None, mode: str
) -> dict[str, str]:
    """In passthrough mode, inject ``x-api-key`` from the env when absent."""
    if mode != "passthrough" or not api_key_env:
        return headers
    if "authorization" in headers or "x-api-key" in headers:
        return headers
    value = os.environ.get(api_key_env)
    if not value:
        raise DispatchError(f"env var {api_key_env!r} is not set")
    return {**headers, "x-api-key": value}


def _resolve_api_key(api_key_env: str | None) -> str | None:
    """Translate-mode helper: read ``api_key_env`` from the environment."""
    if not api_key_env:
        return None
    value = os.environ.get(api_key_env)
    if not value:
        raise DispatchError(f"env var {api_key_env!r} is not set")
    return value
