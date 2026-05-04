"""DI seam factories and the seven endpoint handlers.

The completion callable is injected via FastAPI's dependency system so
tests can swap it out with ``app.dependency_overrides[get_completion]``
(and similarly for the per-shape siblings). Each handler is a thin
wrapper that calls :func:`magos.ingress.http.run.run_endpoint`.

Endpoints:

POST handlers
    /v1/messages                  Anthropic Messages shape
    /v1/messages/count_tokens     Anthropic count_tokens shape
    /v1/chat/completions          OpenAI Chat Completions shape
    /v1/responses                 OpenAI Responses shape

Auxiliary OpenAI Responses handlers (passthrough only — translate mode
rejects non-POST):
    GET    /v1/responses/{id}                   retrieve
    DELETE /v1/responses/{id}                   cancel
    GET    /v1/responses/{id}/input_items       list input items

Match expressions see the **templated** path so rules stay stable across
response IDs; the dispatcher forwards the **concrete** path via
``RoutedRequest.actual_path``.
"""

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
    """Upstream completion for /v1/chat/completions (OpenAI Chat shape)."""
    return cast(CompletionFn, litellm.acompletion)


def get_anthropic_messages_completion() -> CompletionFn:
    """Upstream completion for /v1/messages (Anthropic-unified shape).

    Returns ``_dispatch_anthropic_messages`` rather than ``litellm.
    anthropic_messages`` directly: the LiteLLM helper leaks the
    provider prefix into the outbound body when dispatched to non-
    Anthropic upstreams (OpenRouter rejects ``model: 'openrouter/qwen/
    ...'`` with 400 *not a valid model ID*). The dispatcher detects
    non-Anthropic dispatch and re-routes through ``litellm.acompletion``
    + manual Anthropic↔OpenAI body translation, which strips the
    prefix correctly. Anthropic-bound traffic stays on the fast pass-
    through.
    """
    return cast(CompletionFn, _dispatch_anthropic_messages)


def get_responses_completion() -> CompletionFn:
    """Upstream completion for /v1/responses (litellm's Responses API)."""
    return cast(CompletionFn, litellm.aresponses)


def get_count_tokens_completion() -> CompletionFn:
    """Upstream count-tokens call for /v1/messages/count_tokens.

    LiteLLM's ``acount_tokens`` auto-selects between local tokenizers and
    the provider's native count-tokens endpoint based on the model id.
    """
    return cast(CompletionFn, litellm.acount_tokens)


CompletionDep = Annotated[CompletionFn, Depends(get_completion)]
AnthropicMessagesCompletionDep = Annotated[CompletionFn, Depends(get_anthropic_messages_completion)]
ResponsesCompletionDep = Annotated[CompletionFn, Depends(get_responses_completion)]
CountTokensCompletionDep = Annotated[CompletionFn, Depends(get_count_tokens_completion)]
SettingsDep = Annotated[MagosSettings, Depends(get_settings)]


def register_handlers(app: FastAPI) -> None:
    """Register the seven endpoint handlers on ``app``."""

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
