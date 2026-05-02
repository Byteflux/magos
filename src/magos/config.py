"""Declarative process-level configuration for magos.

``MagosSettings`` covers the small set of knobs that belong in the process
environment: bind address, log/trace setup, and the path to ``magos.yaml``.
Routing-shape decisions (passthrough toggling, count_tokens mode, provider
lookup) live in ``magos.yaml`` and reach the app via ``app.state.routing``;
this module owns only what an operator sets via env or ``.env``.

Example::

    MAGOS_PORT=9000 MAGOS_LOG_JSON=1 MAGOS_CONFIG_PATH=/etc/magos.yaml \\
      python -m magos
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from magos.obs import get_logger

log = get_logger("magos.config")

KompressBackend = Literal["auto", "pytorch"]

# Env vars that have no effect today. ``extra="ignore"`` would silently drop
# them; we warn on startup so an operator with stale ``.env`` files notices
# their intent isn't being applied (the equivalent now lives in ``magos.yaml``).
_REMOVED_ENV_VARS: tuple[str, ...] = (
    "MAGOS_ANTHROPIC_PASSTHROUGH_ENABLED",
    "MAGOS_ANTHROPIC_UPSTREAM_URL",
    "MAGOS_COUNT_TOKENS_PASSTHROUGH_PROVIDERS",
)


class MagosSettings(BaseSettings):
    """Runtime settings for the magos server."""

    model_config = SettingsConfigDict(
        env_prefix="MAGOS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    host: str = Field(default="127.0.0.1", description="HTTP listen host")
    port: int = Field(default=8000, ge=1, le=65535, description="HTTP listen port")

    log_level: str = Field(default="INFO", description="structlog filter level")
    log_json: bool = Field(default=False, description="render structlog as JSON")

    otel_enabled: bool = Field(default=False, description="ship OTLP spans")
    otel_endpoint: str | None = Field(
        default=None, description="OTLP HTTP endpoint; default uses OTel SDK fallback"
    )

    config_path: str = Field(
        default_factory=lambda: str(Path.home() / ".magos" / "magos.yaml"),
        description=(
            "Path to the routing config YAML. Defaults to ~/.magos/magos.yaml; "
            "override with MAGOS_CONFIG_PATH or the --config CLI flag. The file "
            "must exist; ship a copy of magos.example.yaml as a starting point."
        ),
    )

    kompress_backend: KompressBackend = Field(
        default="auto",
        description=(
            "Which backend Headroom's Kompress uses. 'auto' (default) lets "
            "Headroom prefer ONNX Runtime when installed and fall back to "
            "PyTorch. 'pytorch' forces PyTorch (auto-picks CUDA/MPS/CPU); "
            "this is the path to choose for GPU acceleration. Applied "
            "process-wide at FastAPI startup via the lifespan hook."
        ),
    )

    kompress_preload: bool = Field(
        default=True,
        description=(
            "When a routing rule uses the 'compress' rewrite, kick off a "
            "background task at startup that loads Headroom's Kompress "
            "model weights via asyncio.to_thread. Avoids paying multi-second "
            "model-load latency on the first compressed request. Headroom's "
            "internal threading.Lock ensures concurrent compress() calls "
            "block safely until the preload completes. Set to False to "
            "fall back to lazy on-demand loading."
        ),
    )

    access_log: bool = Field(
        default=True,
        description=(
            "Emit one structlog line per HTTP request via uvicorn's access "
            "logger. Set MAGOS_ACCESS_LOG=0 to silence."
        ),
    )

    metrics_enabled: bool = Field(
        default=False,
        description=(
            "Mount a Prometheus-format /metrics endpoint backed by the OTel "
            "MeterProvider. When enabled, the server installs the Prometheus "
            "exporter at startup and exposes registry + future per-subsystem "
            "metrics. Off by default to avoid touching the global "
            "MeterProvider when the operator hasn't asked for it."
        ),
    )


def get_settings() -> MagosSettings:
    """Construct fresh settings from the current environment."""
    settings = MagosSettings()
    _warn_on_removed_env_vars()
    return settings


def _warn_on_removed_env_vars() -> None:
    """Log a warning when an inert env var is still set in the environment."""
    for name in _REMOVED_ENV_VARS:
        if name in os.environ:
            log.warning(
                "config.removed_env_var",
                name=name,
                hint="this env var is ignored; configure the equivalent in magos.yaml",
            )
