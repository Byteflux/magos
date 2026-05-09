"""Structlog setup and the unified logger factory.

`configure_logging` is called once at startup; until it runs,
`get_logger` falls back to structlog defaults. `MAGOS_LOG_JSON=1`
flips the renderer from console to JSON.

Two log levels apply, deliberately split:

  * `level` (env: `MAGOS_LOG_LEVEL`, default INFO) controls `magos.*`
    loggers and the structlog filtering bound logger. This is what
    operators tune when they want more or less detail from our own code.
  * `MAGOS_THIRD_PARTY_LOG_LEVEL` (default ERROR) controls every other
    logger via the root level. The default is a high floor so noise from
    uvicorn, litellm, httpx, transformers, mitmproxy, otel, etc. stays
    silent unless something is actually broken. Raise this to `WARNING`
    or `INFO` for debugging; we never lower it implicitly.

Libraries with logging machinery that bypasses stdlib propagation (LiteLLM,
transformers) get explicit shims aligned with the third-party floor.

## Log-level convention

The `magos.*` loggers follow this convention so operators can tune
verbosity and alerts without inspecting individual call sites:

  * **DEBUG** -- per-step internal mechanics (cache misses, no-op
    transforms, model-limit resolution, registry refresh attempts).
    Off at the default INFO level; flip on for local diagnosis.
  * **INFO** -- significant lifecycle events and per-request milestones
    (`server.ready`, `route.matched`, `dispatch`, `egress.usage`). The
    "operational story" of the proxy. Volume scales with request rate.
  * **WARNING** -- a recoverable problem: the client may have received a
    degraded response, or an internal subsystem failed gracefully.
    Examples: `route.dispatch_error` (returned 503 to client),
    `compress.pipeline_warm_failed` (compression disabled but proxy
    serves), `registry.refresh.failed` (kept old state). Operationally
    noteworthy; not necessarily alert-worthy.
  * **ERROR** -- an unhandled exception escaped a boundary. The proxy
    couldn't fulfill its contract on this request; alert-worthy.
    Reserved for unexpected paths -- expected failure modes (bad config,
    upstream 5xx) belong at WARNING.
  * **EXCEPTION** -- ERROR + traceback, via `log.exception`. Use only
    when the traceback is the diagnosis (e.g. registry adapter raised
    something we didn't anticipate). The Rich traceback formatter is
    configured to omit frame locals to avoid leaking secrets.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from typing import Any, cast

import structlog


def _exception_formatter() -> Any:
    """Return a structlog exception formatter that omits frame locals.

    Defaults to `RichTracebackFormatter(show_locals=False)` when rich is
    importable; otherwise falls back to `plain_traceback`. The locals dump
    is the loud failure mode: it pretty-prints every frame's local variables
    (request payload, headers including API keys, internal SDK state) into
    a multi-page Rich box. Hiding it keeps the traceback compact and avoids
    leaking secrets to logs.
    """
    try:
        return structlog.dev.RichTracebackFormatter(show_locals=False)
    except Exception:
        return structlog.dev.plain_traceback


def _resolve_level(name: str, default: int) -> int:
    """Map a level name to a stdlib logging level int, falling back to `default`."""
    resolved = getattr(logging, name.upper(), None)
    return resolved if isinstance(resolved, int) else default


def configure_logging(level: str = "INFO", *, json: bool | None = None) -> None:
    """Configure structlog and bridge stdlib logging through it."""
    use_json = json if json is not None else os.environ.get("MAGOS_LOG_JSON", "0") == "1"
    magos_level = _resolve_level(level, logging.INFO)
    third_party_level = _resolve_level(
        os.environ.get("MAGOS_THIRD_PARTY_LOG_LEVEL", "ERROR"), logging.ERROR
    )
    if use_json:
        renderer: Any = structlog.processors.JSONRenderer()
        timestamp_fmt = "iso"
    else:
        # Auto-color when stderr is a TTY; allow MAGOS_LOG_COLOR=0/1 to override.
        color_env = os.environ.get("MAGOS_LOG_COLOR")
        colors = (color_env == "1") if color_env is not None else sys.stderr.isatty()
        renderer = structlog.dev.ConsoleRenderer(
            colors=colors,
            force_colors=colors,
            pad_event_to=0,
            sort_keys=False,
            exception_formatter=_exception_formatter(),
        )
        timestamp_fmt = "%H:%M:%S"
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt=timestamp_fmt, utc=False),
    ]
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(magos_level),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers = [handler]
    # Root sits at the third-party floor (default ERROR) so anything we
    # don't own stays silent unless explicitly raised. The `magos` logger
    # is bumped below to `magos_level` so our own structured events flow.
    root.setLevel(third_party_level)
    logging.getLogger("magos").setLevel(magos_level)

    # uvicorn ships its own LOGGING_CONFIG; we pass log_config=None to uvicorn
    # so its loggers default to propagate=True. Reset any prior handlers
    # (e.g. from a previous configure_logging call in tests) and unset the
    # explicit level so they cascade from root (the third-party floor).
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
        lg.setLevel(logging.NOTSET)

    # Silence transformers' weight-load report (emitted when Kompress loads
    # ModernBERT without the LM head). Env var is read at transformers import;
    # the module call covers the case where it's already imported.
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    transformers = sys.modules.get("transformers")
    if transformers is not None:
        with contextlib.suppress(AttributeError):
            transformers.logging.set_verbosity_error()

    # LiteLLM's logging bypasses stdlib propagation: it attaches its own
    # handler at import gated on `LITELLM_LOG`, plus emits hardcoded
    # `print()` banners ("Give Feedback", "Provider List") on error paths.
    # Align the env var (read at import) and the stdlib logger level (covers
    # the already-imported case) with the third-party floor; suppress_debug_info
    # quiets the print banners.
    floor_name = logging.getLevelName(third_party_level)
    os.environ.setdefault("LITELLM_LOG", floor_name)
    logging.getLogger("LiteLLM").setLevel(third_party_level)
    litellm = sys.modules.get("litellm")
    if litellm is not None:
        with contextlib.suppress(AttributeError):
            litellm.suppress_debug_info = True  # type: ignore[attr-defined]


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger; safe to call before `configure_logging`."""
    return cast(
        structlog.stdlib.BoundLogger,
        structlog.get_logger(name) if name else structlog.get_logger(),
    )
