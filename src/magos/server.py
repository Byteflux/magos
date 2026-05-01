"""FastAPI server for the magos LLM proxy.

Exposes two non-streaming endpoints:

- ``POST /v1/messages``           Anthropic Messages shape
- ``POST /v1/chat/completions``   OpenAI Chat Completions shape

The upstream completion callable is injected via FastAPI's dependency system
so tests can swap it out with ``app.dependency_overrides[get_completion]``.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, cast

import litellm
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
from starlette.datastructures import Headers

from magos import __version__
from magos.config import MagosSettings, get_settings
from magos.obs import get_logger
from magos.passthrough import (
    call_anthropic_passthrough,
    should_anthropic_passthrough,
    stream_anthropic_passthrough,
)
from magos.proxy import (
    proxy_anthropic_messages,
    proxy_openai_chat_completions,
    stream_anthropic_messages,
    stream_openai_chat_completions,
)
from magos.tokens import count_input_tokens

log = get_logger("magos.server")

CompletionFn = Callable[..., Awaitable[Any]]


def get_completion() -> CompletionFn:
    """Dependency-injection seam for the upstream completion callable."""
    return cast(CompletionFn, litellm.acompletion)


CompletionDep = Annotated[CompletionFn, Depends(get_completion)]
SettingsDep = Annotated[MagosSettings, Depends(get_settings)]


# Hop-by-hop headers (RFC 7230) plus a few that httpx must own. Everything
# else is forwarded so upstream sees the client's auth, version pins, and
# beta flags verbatim, which preserves provider billing shape.
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


def _forwardable_headers(headers: Headers) -> dict[str, str]:
    """Return inbound headers minus hop-by-hop and content-shaping ones."""
    return {k: v for k, v in headers.items() if k.lower() not in _BLOCKED_FORWARD_HEADERS}


def create_app() -> FastAPI:
    app = FastAPI(title="magos", version=__version__)

    @app.post("/v1/messages")
    async def anthropic_messages(
        request: Request,
        completion: CompletionDep,
        settings: SettingsDep,
    ) -> Any:
        forward = _forwardable_headers(request.headers)
        raw_body = await request.body()
        try:
            body: dict[str, Any] = json.loads(raw_body) if raw_body else {}
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}") from exc

        # Same-shape Anthropic-in / Anthropic-upstream is forwarded verbatim
        # (raw bytes; no re-serialisation) so OAuth bearers, anthropic-beta
        # flags, version pins, and cache_control byte boundaries all reach
        # the provider untouched. LiteLLM short-circuits on missing api_key
        # and cannot carry an OAuth bearer; reserialised JSON also breaks
        # prompt caching, billing the request as fresh long-context input.
        if settings.anthropic_passthrough_enabled and should_anthropic_passthrough(body):
            model = str(body.get("model", ""))
            if body.get("stream") is True:
                return StreamingResponse(
                    stream_anthropic_passthrough(
                        raw_body, forward, settings.anthropic_upstream_url, model_hint=model
                    ),
                    media_type="text/event-stream",
                )
            status, raw, content_type = await call_anthropic_passthrough(
                raw_body, forward, settings.anthropic_upstream_url, model_hint=model
            )
            return Response(content=raw, status_code=status, media_type=content_type)

        if body.get("stream") is True:
            try:
                stream = stream_anthropic_messages(
                    body, completion=completion, forward_headers=forward
                )
            except ValidationError as exc:
                raise HTTPException(status_code=400, detail=exc.errors()) from exc
            return StreamingResponse(stream, media_type="text/event-stream")
        try:
            return await proxy_anthropic_messages(
                body, completion=completion, forward_headers=forward
            )
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=exc.errors()) from exc
        except HTTPException:
            raise
        except Exception as exc:
            log.error(
                "upstream_failure",
                endpoint="messages",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            raise HTTPException(status_code=502, detail=f"upstream failure: {exc}") from exc

    @app.post("/v1/messages/count_tokens")
    async def anthropic_count_tokens(
        body: dict[str, Any],
        request: Request,
        settings: SettingsDep,
    ) -> dict[str, int]:
        forward = _forwardable_headers(request.headers)
        try:
            n = await count_input_tokens(
                body,
                passthrough_providers=settings.count_tokens_passthrough_providers,
                forward_headers=forward,
            )
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=exc.errors()) from exc
        return {"input_tokens": n}

    @app.post("/v1/chat/completions")
    async def openai_chat_completions(
        body: dict[str, Any],
        request: Request,
        completion: CompletionDep,
    ) -> Any:
        forward = _forwardable_headers(request.headers)
        if body.get("stream") is True:
            return StreamingResponse(
                stream_openai_chat_completions(
                    body, completion=completion, forward_headers=forward
                ),
                media_type="text/event-stream",
            )
        try:
            return await proxy_openai_chat_completions(
                body, completion=completion, forward_headers=forward
            )
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=exc.errors()) from exc
        except HTTPException:
            raise
        except Exception as exc:
            log.error(
                "upstream_failure",
                endpoint="chat_completions",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            raise HTTPException(status_code=502, detail=f"upstream failure: {exc}") from exc

    return app


app = create_app()
