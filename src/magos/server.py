"""FastAPI server for the magos LLM proxy.

Four endpoints, all routed through the declarative rules in ``magos.yaml``:

- ``POST /v1/messages``               Anthropic Messages shape
- ``POST /v1/messages/count_tokens``  Anthropic count_tokens shape
- ``POST /v1/chat/completions``       OpenAI Chat Completions shape
- ``POST /v1/responses``              OpenAI Responses shape

Each handler parses the inbound body, builds a ``RoutedRequest``, calls
``route()`` to pick a rule, and hands the resulting ``RouteDecision`` to
``dispatch_decision()``. ``RouteError`` outcomes (404 unmatched, 503
dispatch error) are rendered through the per-endpoint error envelope so
clients see a familiar shape on routing-layer failures.

The completion callable is injected via FastAPI's dependency system so
tests can swap it out with ``app.dependency_overrides[get_completion]``.
The routing config lives on ``app.state.routing`` so tests can replace it
directly without re-running ``create_app``.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, cast

import litellm
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError
from starlette.datastructures import Headers

from magos import __version__
from magos.config import MagosSettings, get_settings
from magos.obs import get_logger
from magos.routing import (
    Endpoint,
    RoutedRequest,
    RouteError,
    RoutingConfig,
    error_envelope,
    format_dispatch_error_message,
    load_config,
    route,
)
from magos.routing.dispatch import DispatchError, dispatch_decision

log = get_logger("magos.server")

CompletionFn = Callable[..., Awaitable[Any]]


def get_completion() -> CompletionFn:
    """Upstream completion for /v1/chat/completions (OpenAI Chat shape)."""
    return cast(CompletionFn, litellm.acompletion)


def get_anthropic_messages_completion() -> CompletionFn:
    """Upstream completion for /v1/messages (Anthropic-unified shape).

    LiteLLM's ``anthropic_messages`` accepts Anthropic-shape requests and
    emits Anthropic-shape responses regardless of upstream provider, so it
    is the right call site for both Anthropic-on-Anthropic and cross-
    provider routing (Anthropic shape -> OpenAI/Gemini/Bedrock/etc.).
    """
    return cast(CompletionFn, litellm.anthropic_messages)


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
    """Return inbound headers minus hop-by-hop and content-shaping ones.

    Keys are lowercased so routing matchers and rewrites can use case-
    insensitive lookups uniformly.
    """
    return {k.lower(): v for k, v in headers.items() if k.lower() not in _BLOCKED_FORWARD_HEADERS}


def create_app(routing: RoutingConfig | None = None) -> FastAPI:
    """Build the FastAPI app, loading routing config from disk by default.

    Tests can pass ``routing`` directly to skip the YAML round-trip; in
    that case ``MAGOS_CONFIG_PATH`` is ignored.
    """
    settings = MagosSettings()
    cfg = routing if routing is not None else load_config(settings.config_path)
    app = FastAPI(title="magos", version=__version__)
    app.state.routing = cfg

    @app.post("/v1/messages")
    async def anthropic_messages(  # type: ignore[unused-ignore]
        request: Request, completion: AnthropicMessagesCompletionDep
    ) -> Any:
        return await _run("/v1/messages", request, completion)

    @app.post("/v1/messages/count_tokens")
    async def anthropic_count_tokens(  # type: ignore[unused-ignore]
        request: Request, completion: CountTokensCompletionDep
    ) -> Any:
        return await _run("/v1/messages/count_tokens", request, completion)

    @app.post("/v1/chat/completions")
    async def openai_chat_completions(  # type: ignore[unused-ignore]
        request: Request, completion: CompletionDep
    ) -> Any:
        return await _run("/v1/chat/completions", request, completion)

    @app.post("/v1/responses")
    async def openai_responses(  # type: ignore[unused-ignore]
        request: Request, completion: ResponsesCompletionDep
    ) -> Any:
        return await _run("/v1/responses", request, completion)

    # Auxiliary /v1/responses endpoints (passthrough-only): retrieve, cancel,
    # list input items. Match expressions see the templated path so rules
    # stay stable across response IDs; the dispatcher forwards the concrete
    # path via ``RoutedRequest.actual_path``.
    @app.get("/v1/responses/{response_id}")
    async def retrieve_response(  # type: ignore[unused-ignore]
        request: Request, response_id: str, completion: ResponsesCompletionDep
    ) -> Any:
        return await _run(
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
        return await _run(
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
        return await _run(
            "/v1/responses/{id}/input_items",
            request,
            completion,
            method="GET",
            actual_path=f"/v1/responses/{response_id}/input_items",
        )

    return app


async def _run(
    endpoint: Endpoint,
    request: Request,
    completion: CompletionFn,
    *,
    method: str = "POST",
    actual_path: str | None = None,
) -> Response | StreamingResponse | dict[str, Any]:
    """Shared routing + dispatch flow used by every handler."""
    raw_body = await request.body()
    try:
        body: dict[str, Any] = json.loads(raw_body) if raw_body else {}
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="request body must be a JSON object")

    forward = _forwardable_headers(request.headers)
    routed = RoutedRequest(
        endpoint=endpoint,
        headers=forward,
        body=body,
        raw_body=raw_body,
        method=cast(Any, method),
        actual_path=actual_path,
    )
    cfg = cast(RoutingConfig, request.app.state.routing)
    decision_or_err = route(routed, cfg)

    if isinstance(decision_or_err, RouteError):
        return _render_route_error(decision_or_err)

    log.info(
        "route.matched",
        rule=decision_or_err.rule_label(),
        endpoint=endpoint,
        model=str(routed.body.get("model", "")),
        mode=decision_or_err.action.mode,
    )

    try:
        return await dispatch_decision(decision_or_err, completion=completion)
    except DispatchError as exc:
        log.warning(
            "route.dispatch_error",
            rule=decision_or_err.rule_label(),
            endpoint=endpoint,
            error=str(exc),
        )
        err = RouteError(
            status=503,
            code="dispatch_error",
            message=format_dispatch_error_message(str(exc)),
            model=str(routed.body.get("model", "")),
            endpoint=endpoint,
        )
        return _render_route_error(err)
    except ValidationError as exc:
        # Translation-layer schema check rejected the body; surface as 400.
        raise HTTPException(status_code=400, detail=exc.errors()) from exc
    except HTTPException:
        raise
    except Exception as exc:
        log.error(
            "upstream_failure",
            endpoint=endpoint,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        raise HTTPException(status_code=502, detail=f"upstream failure: {exc}") from exc


def _render_route_error(err: RouteError) -> JSONResponse:
    log.info(
        "route." + ("unmatched" if err.code == "unmatched" else "dispatch_error"),
        endpoint=err.endpoint,
        model=err.model,
        message=err.message,
    )
    body = error_envelope(endpoint=err.endpoint, code=err.code, message=err.message)
    return JSONResponse(status_code=err.status, content=body)
