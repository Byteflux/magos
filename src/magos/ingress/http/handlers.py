"""DI seams + data-driven endpoint registration.

``ENDPOINT_TABLE`` drives :func:`register_handlers`; each row is
``(method, path_pattern, template_endpoint, completion_dep_name)``.
Adding a new endpoint means one new row, not a new hand-coded handler.

Match expressions see the *template* endpoint (e.g. ``/v1/responses/{id}``);
the concrete path is forwarded via ``RoutedRequest.actual_path``.
"""

from __future__ import annotations

from typing import Annotated, Any, cast

import litellm
from fastapi import Depends, FastAPI, Request

from magos.config.settings import MagosSettings, get_settings
from magos.egress import CompletionFn
from magos.egress.translate.anthropic import _dispatch_anthropic_messages
from magos.ingress.http.run import run_endpoint
from magos.routing import Endpoint


def get_completion() -> CompletionFn:
    return cast(CompletionFn, litellm.acompletion)


def get_anthropic_messages_completion() -> CompletionFn:
    """Upstream for /v1/messages.

    ``litellm.anthropic_messages`` leaks the provider prefix into the
    outbound body for non-Anthropic upstreams (OpenRouter rejects
    ``model: 'openrouter/qwen/...'``); the dispatcher re-routes those
    through ``litellm.acompletion`` + body translation.
    """
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

# Each row: (http_method, path_pattern, template_endpoint, completion_dep)
# ``completion_dep`` is the Depends-annotated type alias for the right upstream.
ENDPOINT_TABLE: list[tuple[str, str, str, Any]] = [
    ("POST", "/v1/messages", "/v1/messages", AnthropicMessagesCompletionDep),
    ("POST", "/v1/messages/count_tokens", "/v1/messages/count_tokens", CountTokensCompletionDep),
    ("POST", "/v1/chat/completions", "/v1/chat/completions", CompletionDep),
    ("POST", "/v1/responses", "/v1/responses", ResponsesCompletionDep),
    ("GET", "/v1/responses/{response_id}", "/v1/responses/{id}", ResponsesCompletionDep),
    ("DELETE", "/v1/responses/{response_id}", "/v1/responses/{id}", ResponsesCompletionDep),
    (
        "GET",
        "/v1/responses/{response_id}/input_items",
        "/v1/responses/{id}/input_items",
        ResponsesCompletionDep,
    ),
]


def register_handlers(app: FastAPI) -> None:
    """Register all LLM proxy endpoints from :data:`ENDPOINT_TABLE`."""
    for http_method, path_pattern, template_endpoint, dep_type in ENDPOINT_TABLE:
        _register_one(app, http_method, path_pattern, cast(Endpoint, template_endpoint), dep_type)


def _register_one(
    app: FastAPI,
    http_method: str,
    path_pattern: str,
    template_endpoint: Endpoint,
    dep_type: type,
) -> None:
    _te: Endpoint = template_endpoint  # captured by closure; not a FastAPI parameter.

    async def _handler(  # type: ignore[unused-ignore]
        request: Request,
        completion: CompletionFn,
    ) -> Any:
        # ``request.url.path`` and ``request.method`` carry through both
        # templated (``/v1/responses/resp_abc``) and non-templated paths;
        # for the latter ``actual_path`` matches ``_te`` so it's a no-op.
        return await run_endpoint(
            _te, request, completion, method=request.method, actual_path=request.url.path
        )

    # Replace the static ``CompletionFn`` annotation with the Depends-annotated
    # type alias so FastAPI injects the right upstream callable per endpoint.
    _handler.__annotations__["completion"] = dep_type
    _handler.__name__ = f"{http_method.lower()}_{path_pattern.replace('/', '_').strip('_')}"
    app.add_api_route(path_pattern, _handler, methods=[http_method])
