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

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, cast

import litellm
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError
from starlette.datastructures import Headers

from magos import __version__
from magos.config import MagosSettings, get_settings
from magos.config_loader import load_full_config
from magos.obs import get_logger
from magos.registry.refresher import Refresher
from magos.registry.schema import RegistryYaml
from magos.routing import (
    Compress,
    Endpoint,
    RoutedRequest,
    RouteError,
    RoutingConfig,
    error_envelope,
    format_dispatch_error_message,
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


def _mount_admin_registry_endpoints(app: FastAPI) -> None:
    """Expose registry inspection + force-refresh under ``/admin/registry``.

    Mounted only when a ``Refresher`` is active. The CLI uses these to
    show server-state and trigger out-of-band refreshes; ``list`` /
    ``show`` fall back to disk when the server is down.
    """
    from magos.registry.discovery.base import DiscoveryError  # noqa: PLC0415
    from magos.registry.store import serialize  # noqa: PLC0415

    @app.get("/admin/registry", include_in_schema=False)
    async def get_registry(request: Request) -> Response:
        refresher = cast(Refresher, request.app.state.refresher)
        return Response(content=serialize(refresher.state), media_type="application/json")

    @app.post("/admin/registry/refresh", include_in_schema=False)
    async def refresh_registry(request: Request, provider: str | None = None) -> Response:
        refresher = cast(Refresher, request.app.state.refresher)
        registry_cfg = cast(RegistryYaml, request.app.state.registry_config)
        targets = [provider] if provider else list(registry_cfg.providers)
        unknown = [p for p in targets if p not in registry_cfg.providers]
        if unknown:
            raise HTTPException(
                status_code=404, detail=f"unknown provider(s): {', '.join(unknown)}"
            )
        refreshed: list[str] = []
        failed: dict[str, str] = {}
        for name in targets:
            try:
                await refresher.refresh(name)
                refreshed.append(name)
            except DiscoveryError as exc:
                failed[name] = str(exc)
        return JSONResponse({"refreshed": refreshed, "failed": failed})

    @app.post("/admin/registry/prune", include_in_schema=False)
    async def prune_registry(request: Request) -> Response:
        """Trigger a prune by refreshing every provider.

        The deprecation state machine drops past-grace entries on every
        successful refresh, so a full refresh round is the simplest way
        to surface the operator-visible "prune" action without adding
        a separate code path.
        """
        refresher = cast(Refresher, request.app.state.refresher)
        registry_cfg = cast(RegistryYaml, request.app.state.registry_config)
        before = sum(1 for e in refresher.state.entries.values() if e.is_deprecated)
        for name in registry_cfg.providers:
            try:
                await refresher.refresh(name)
            except DiscoveryError:
                continue  # other providers still get their chance
        after = sum(1 for e in refresher.state.entries.values() if e.is_deprecated)
        return JSONResponse({"deprecated_before": before, "deprecated_after": after})


def _mount_metrics_endpoint(app: FastAPI) -> None:
    """Expose Prometheus-format metrics at ``GET /metrics``.

    ``prometheus_client``'s default ``REGISTRY`` is what the OTel
    PrometheusMetricReader writes into, so generating the text export
    here returns whatever the OTel meters have produced.
    """
    try:
        from prometheus_client import (  # noqa: PLC0415
            CONTENT_TYPE_LATEST,
            REGISTRY,
            generate_latest,
        )
    except ImportError:
        log.warning("metrics.endpoint_skipped", reason="prometheus_client missing")
        return

    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint() -> Response:
        return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


def _resolve_models_path(config_path: str, registry_cfg: RegistryYaml) -> Path:
    """Anchor the registry block's ``models_path`` to the config file's parent.

    Delegates to ``magos.config_loader.resolve_models_path`` so server
    boot, CLI ``list --from-disk``, and CLI ``show`` all agree on the
    same file regardless of CWD. ``models.json`` is server-owned: out-
    of-process readers are fine; the only writer is the Refresher.
    """
    from magos.config_loader import resolve_models_path  # noqa: PLC0415

    return resolve_models_path(config_path, registry_cfg)


def _config_uses_compress(cfg: RoutingConfig) -> bool:
    """True iff any rewrite (pre or per-rule) is a Compress."""
    if any(isinstance(rw, Compress) for rw in cfg.pre_rewrites):
        return True
    return any(isinstance(rw, Compress) for rule in cfg.rules for rw in rule.rewrites)


def _configure_metrics_provider() -> None:
    """Install a global OTel MeterProvider with the Prometheus exporter.

    Idempotent in practice: ``set_meter_provider`` only honors the first
    real provider per process, so re-invocation logs a warning and is a
    no-op. ``prometheus_client.start_http_server`` is intentionally
    avoided — we let the FastAPI mount expose ``/metrics`` so the server
    binds one port for everything.
    """
    try:
        from opentelemetry import metrics  # noqa: PLC0415
        from opentelemetry.exporter.prometheus import PrometheusMetricReader  # noqa: PLC0415
        from opentelemetry.sdk.metrics import MeterProvider  # noqa: PLC0415
    except ImportError as exc:
        log.warning(
            "metrics.exporter_unavailable",
            error=str(exc),
            hint="install opentelemetry-exporter-prometheus to enable /metrics",
        )
        return

    reader = PrometheusMetricReader()
    metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))
    log.info("metrics.provider_configured", exporter="prometheus")


def _force_kompress_pytorch() -> None:
    """Make Headroom's Kompress loader skip the ONNX path.

    Headroom's ``_load_kompress`` checks ``_is_onnx_available()`` from the
    module namespace at call time and prefers ONNX when both backends are
    installed. Replacing that name with a False-returning stub flips the
    loader to the PyTorch branch (``_load_kompress_pytorch``), which
    auto-selects CUDA/MPS/CPU via ``device='auto'``. No Headroom patch
    needed — Python late-binding does the work.

    Silently no-ops if Kompress isn't importable (no compress rules, or
    deps missing).
    """
    try:
        from headroom.transforms import kompress_compressor  # noqa: PLC0415
    except Exception as exc:
        log.warning(
            "compress.kompress_force_pytorch_skipped",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return

    kompress_compressor._is_onnx_available = lambda: False
    log.info("compress.kompress_backend_forced", backend="pytorch")


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Warm Headroom, apply the Kompress override, and start the registry.

    Headroom builds a thread-locked singleton on first call (tokenizer +
    transform pipeline init). Pulling that cost into startup avoids
    burying multi-second latency in the first user request. The registry
    Refresher is started here when ``providers:`` is non-empty; for
    test-only configs without a registry, no background task runs.
    """
    settings = MagosSettings()
    if settings.kompress_backend == "pytorch":
        _force_kompress_pytorch()
    if settings.metrics_enabled:
        _configure_metrics_provider()

    cfg = cast(RoutingConfig, app.state.routing)
    preload_task: asyncio.Task[None] | None = None
    if _config_uses_compress(cfg):
        try:
            from headroom.compress import _get_pipeline  # noqa: PLC0415

            _get_pipeline()
            log.info("compress.pipeline_warmed")
        except Exception as exc:
            log.warning(
                "compress.pipeline_warm_failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )
        if settings.kompress_preload:
            preload_task = asyncio.create_task(
                _preload_kompress_model(), name="magos.kompress.preload"
            )

    refresher: Refresher | None = cast(Refresher | None, app.state.refresher)
    if refresher is not None:
        await refresher.start()
        log.info("registry.refresher.started", providers=list(refresher._config.providers))
    log.info(
        "server.ready",
        version=__version__,
        rules=len(cfg.rules),
        metrics=settings.metrics_enabled,
    )
    try:
        yield
    finally:
        log.info("server.shutting_down")
        if preload_task is not None and not preload_task.done():
            preload_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await preload_task
        if refresher is not None:
            await refresher.stop()
            log.info("registry.refresher.stopped")


async def _preload_kompress_model() -> None:
    """Warm Kompress model weights off the event loop.

    Headroom's ``_load_kompress`` is a thread-locked, double-checked
    singleton populator (see ``_kompress_cache``); a request that races
    in via ``compress()`` blocks on the same lock and reuses the cached
    model. The leading underscore is a stability risk: a Headroom
    version bump may rename it, so ImportError falls back to lazy load.
    """
    try:
        from headroom.transforms.kompress_compressor import (  # noqa: PLC0415
            HF_MODEL_ID,
            _load_kompress,
        )
    except ImportError as exc:
        log.warning("compress.kompress_preload_unavailable", error=str(exc))
        return
    log.info("compress.kompress_preload_started", model=HF_MODEL_ID)
    started = time.perf_counter()
    try:
        await asyncio.to_thread(_load_kompress, HF_MODEL_ID, "auto")
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        log.info("compress.kompress_warmed", model=HF_MODEL_ID, elapsed_ms=elapsed_ms)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        log.warning(
            "compress.kompress_warm_failed",
            error=str(exc),
            error_type=type(exc).__name__,
            elapsed_ms=elapsed_ms,
        )


def create_app(
    routing: RoutingConfig | None = None,
    *,
    registry: RegistryYaml | None = None,
) -> FastAPI:
    """Build the FastAPI app, loading routing + registry config from disk.

    Tests can pass ``routing`` (and optionally ``registry``) directly to
    skip the YAML round-trip; in that case ``MAGOS_CONFIG_PATH`` is
    ignored. When ``routing`` is omitted, both halves are parsed from
    ``MAGOS_CONFIG_PATH`` via :func:`load_full_config`.

    A ``Refresher`` is constructed when the registry block declares any
    providers; otherwise the registry feature is dormant and existing
    routing rules behave exactly as before.
    """
    settings = MagosSettings()
    if routing is None:
        full = load_full_config(settings.config_path)
        cfg = full.routing
        registry_cfg = registry if registry is not None else full.registry
    else:
        cfg = routing
        registry_cfg = registry if registry is not None else RegistryYaml()

    app = FastAPI(title="magos", version=__version__, lifespan=_lifespan)
    app.state.routing = cfg
    app.state.registry_config = registry_cfg
    app.state.refresher = (
        Refresher(registry_cfg, _resolve_models_path(settings.config_path, registry_cfg))
        if registry_cfg.providers
        else None
    )

    if settings.metrics_enabled:
        _mount_metrics_endpoint(app)
    if app.state.refresher is not None:
        _mount_admin_registry_endpoints(app)

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
    refresher = cast("Refresher | None", request.app.state.refresher)
    registry_cfg = cast(RegistryYaml, request.app.state.registry_config)
    decision_or_err = route(
        routed,
        cfg,
        registry=refresher.state if refresher is not None else None,
        registry_settings=registry_cfg.registry if refresher is not None else None,
        providers=registry_cfg.providers if refresher is not None else None,
    )

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
