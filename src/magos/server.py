"""FastAPI server for the magos LLM proxy.

Exposes two non-streaming endpoints:

- ``POST /v1/messages``           Anthropic Messages shape
- ``POST /v1/chat/completions``   OpenAI Chat Completions shape

The upstream completion callable is injected via FastAPI's dependency system
so tests can swap it out with ``app.dependency_overrides[get_completion]``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated, Any, cast

import litellm
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from magos import __version__
from magos.obs import get_logger
from magos.proxy import (
    proxy_anthropic_messages,
    proxy_openai_chat_completions,
    stream_openai_chat_completions,
)

log = get_logger("magos.server")

CompletionFn = Callable[..., Awaitable[Any]]


def get_completion() -> CompletionFn:
    """Dependency-injection seam for the upstream completion callable."""
    return cast(CompletionFn, litellm.acompletion)


CompletionDep = Annotated[CompletionFn, Depends(get_completion)]


def _reject_streaming(body: dict[str, Any]) -> None:
    if body.get("stream") is True:
        raise HTTPException(status_code=501, detail="streaming not yet implemented")


def create_app() -> FastAPI:
    app = FastAPI(title="magos", version=__version__)

    @app.post("/v1/messages")
    async def anthropic_messages(
        body: dict[str, Any],
        completion: CompletionDep,
    ) -> dict[str, Any]:
        _reject_streaming(body)
        try:
            return await proxy_anthropic_messages(body, completion=completion)
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

    @app.post("/v1/chat/completions")
    async def openai_chat_completions(
        body: dict[str, Any],
        completion: CompletionDep,
    ) -> Any:
        if body.get("stream") is True:
            return StreamingResponse(
                stream_openai_chat_completions(body, completion=completion),
                media_type="text/event-stream",
            )
        try:
            return await proxy_openai_chat_completions(body, completion=completion)
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
