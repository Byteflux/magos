"""Structlog setup and the unified logger factory.

``configure_logging`` is called once at startup; until it runs,
``get_logger`` falls back to structlog defaults. ``MAGOS_LOG_JSON=1``
flips the renderer from console to JSON.

The stdlib ``logging`` import is unambiguous here despite the module
name match: Python 3 uses absolute imports, so ``import logging`` from
inside this file resolves to the stdlib package, not back to itself.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from typing import Any, cast

import structlog


def configure_logging(level: str = "INFO", *, json: bool | None = None) -> None:
    """Configure structlog and bridge stdlib logging through it.

    Anything emitted via ``logging`` (uvicorn startup, access logs, third-party
    libraries) is rendered with the same processors as native structlog calls,
    so the operator sees one consistent stream.
    """
    use_json = json if json is not None else os.environ.get("MAGOS_LOG_JSON", "0") == "1"
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
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
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
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # uvicorn ships its own LOGGING_CONFIG; we pass log_config=None to uvicorn
    # so its loggers default to propagate=True. Reset any prior handlers
    # (e.g. from a previous configure_logging call in tests) to be sure.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True

    # Silence transformers' weight-load report (emitted when Kompress loads
    # ModernBERT without the LM head). Env var is read at transformers import;
    # the module call covers the case where it's already imported.
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    transformers = sys.modules.get("transformers")
    if transformers is not None:
        with contextlib.suppress(AttributeError):
            transformers.logging.set_verbosity_error()

    # Silence LiteLLM's chatty INFO logs (echoed twice — once via stdlib
    # logging, once through our structlog bridge) and its hardcoded
    # "Give Feedback" / "Provider List" print() banners on error paths.
    # Env var is read at LiteLLM import; the module-level attr covers
    # the already-imported case.
    os.environ.setdefault("LITELLM_LOG", "WARNING")
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    litellm = sys.modules.get("litellm")
    if litellm is not None:
        with contextlib.suppress(AttributeError):
            litellm.suppress_debug_info = True  # type: ignore[attr-defined]


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger; safe before configure_logging runs."""
    return cast(
        structlog.stdlib.BoundLogger,
        structlog.get_logger(name) if name else structlog.get_logger(),
    )
