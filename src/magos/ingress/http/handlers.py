"""DI seams + endpoint handlers; thin wrappers around
:func:`magos.ingress.http.run.run_endpoint`. The upstream completion
callable is injected via ``Depends`` so tests can override it. Match
expressions see the templated path; concrete path goes via
``RoutedRequest.actual_path``."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated, Any, cast

import litellm
from fastapi import Depends, FastAPI, Request

from magos.config.settings import MagosSettings, get_settings
from magos.egress.translate.anthropic import _dispatch_anthropic_messages
from magos.ingress.http.run import run_endpoint

CompletionFn = Callable[..., Awaitable[Any]]


def get_completion() -> CompletionFn:
    return cast(CompletionFn, litellm.acompletion)


def get_anthropic_messages_completion() -> CompletionFn:
    """Upstream for /v1/messages. ``litellm.anthropic_messages`` leaks
    the provider prefix into the outbound body for non-Anthropic upstreams
    (OpenRouter rejects ``model: 'openrouter/qwen/...'``); the dispatcher
    re-routes those through ``litellm.acompletion`` + body translation."""
    return cast(CompletionFn, _dispatch_anthropic_messages)


def get_responses_completion() -> CompletionFn:
    return cast(CompletionFn, litellm.aresponses)


def get_count_tokens_completion() -> CompletionFn:
    return cast(CompletionFn, litellm.acount_tokens)


CompletionDep = Annotated[CompletionFn, Depends(get_completion)]
AnthropicMessagesCompletionDep = Annotated[CompletionFn, Depends(get_anthropic_messages_completion)]
ResponsesCompletionDep = Annotated[CompletionFn, Depends(get_responses_completion)]
CountTokensCompletionDep = Annotated[CompletionFn, Depends(get_count_tokens_completion)]
SettingsDep = Annotated[MagosSettings, Depends(get_settings)]


def register_handlers(app: FastAPI) -> None:

    @app.post("/v1/messages")
    async def anthropic_messages(  # type: ignore[unused-ignore]
        request: Request, completion: AnthropicMessagesCompletionDep
    ) -> Any:
        return await run_endpoint("/v1/messages", request, completion)

    @app.post("/v1/messages/count_tokens")
    async def anthropic_count_tokens(  # type: ignore[unused-ignore]
        request: Request, completion: CountTokensCompletionDep
    ) -> Any:
        return await run_endpoint("/v1/messages/count_tokens", request, completion)

    @app.post("/v1/chat/completions")
    async def openai_chat_completions(  # type: ignore[unused-ignore]
        request: Request, completion: CompletionDep
    ) -> Any:
        return await run_endpoint("/v1/chat/completions", request, completion)

    @app.post("/v1/responses")
    async def openai_responses(  # type: ignore[unused-ignore]
        request: Request, completion: ResponsesCompletionDep
    ) -> Any:
        return await run_endpoint("/v1/responses", request, completion)

    @app.get("/v1/responses/{response_id}")
    async def retrieve_response(  # type: ignore[unused-ignore]
        request: Request, response_id: str, completion: ResponsesCompletionDep
    ) -> Any:
        return await run_endpoint(
            "/v1/responses/{id}",
            request,
            completion,
            method="GET",
            actual_path=f"/v1/responses/{response_id}",
        )

    @app.delete("/v1/responses/{response_id}")
    async def cancel_response(  # type: ignore[unused-ignore]
        request: Request, response_id: str, completion: ResponsesCompletionDep
    ) -> Any:
        return await run_endpoint(
            "/v1/responses/{id}",
            request,
            completion,
            method="DELETE",
            actual_path=f"/v1/responses/{response_id}",
        )

    @app.get("/v1/responses/{response_id}/input_items")
    async def list_response_input_items(  # type: ignore[unused-ignore]
        request: Request, response_id: str, completion: ResponsesCompletionDep
    ) -> Any:
        return await run_endpoint(
            "/v1/responses/{id}/input_items",
            request,
            completion,
            method="GET",
            actual_path=f"/v1/responses/{response_id}/input_items",
        )
