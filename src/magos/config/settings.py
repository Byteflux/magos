"""Declarative process-level configuration for magos.

``MagosSettings`` covers the small set of knobs that belong in the process
environment: bind address, log/trace setup, and the path to ``magos.yaml``.
Routing-shape decisions (passthrough toggling, count_tokens mode, provider
lookup) live in ``magos.yaml`` and reach the app via ``app.state.routing``;
this module owns only what an operator sets via env or ``.env``.

``MAGOS_HOME`` is a bootstrap-only env var (no settings field): it anchors
the defaults for ``MAGOS_CONFIG_PATH`` and the registry's ``models.json``
path, and is the directory that relative ``registry.models_path`` values
resolve against. Defaults to ``~/.magos`` when unset.

Example::

    MAGOS_PORT=9000 MAGOS_LOG_JSON=1 MAGOS_HOME=/srv/magos \\
      python -m magos
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

KompressBackend = Literal["auto", "pytorch"]


def magos_home() -> Path:
    """Return the magos data directory (``MAGOS_HOME`` or ``~/.magos``).

    Bootstrap-only env var: not a ``MagosSettings`` field. Read directly
    from the environment so the result is consistent across the
    ``config_path`` default factory, ``resolve_models_path`` defaults,
    and any future caller that needs the data directory anchor.
    """
    raw = os.environ.get("MAGOS_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".magos"


class MagosSettings(BaseSettings):
    """Runtime settings for the magos server."""

    model_config = SettingsConfigDict(
        env_prefix="MAGOS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    host: str | None = Field(
        default=None,
        description=(
            "HTTP listen host override. When unset, falls back to "
            "``ingress.http.host`` in magos.yaml (which itself defaults to 127.0.0.1)."
        ),
    )
    port: int | None = Field(
        default=None,
        ge=1,
        le=65535,
        description=(
            "HTTP listen port override. When unset, falls back to "
            "``ingress.http.port`` in magos.yaml (which itself defaults to 8000)."
        ),
    )

    mitm_enabled: bool | None = Field(
        default=None,
        description=(
            "Enable the embedded mitmproxy HTTPS_PROXY listener. When unset, "
            "falls back to ``ingress.mitm.enabled`` in magos.yaml (which "
            "defaults to false)."
        ),
    )
    mitm_host: str | None = Field(
        default=None,
        description=(
            "mitmproxy listener host override. When unset, falls back to "
            "``ingress.mitm.host`` in magos.yaml (which defaults to 127.0.0.1)."
        ),
    )
    mitm_port: int | None = Field(
        default=None,
        ge=1,
        le=65535,
        description=(
            "mitmproxy listener port override. When unset, falls back to "
            "``ingress.mitm.port`` in magos.yaml (which defaults to 8080)."
        ),
    )
    mitm_intercept_hosts: tuple[str, ...] | None = Field(
        default=None,
        description=(
            "Comma-separated list of hosts (and their subdomains) the "
            "mitmproxy ingress should TLS-terminate and route through magos. "
            "When unset, falls back to ``ingress.mitm.intercept_hosts`` in "
            "magos.yaml. Empty string yields an empty tuple."
        ),
    )

    @field_validator("mitm_intercept_hosts", mode="before")
    @classmethod
    def _split_intercept_hosts(cls, v: object) -> object:
        """Allow comma-separated env strings (``MAGOS_MITM_INTERCEPT_HOSTS=a.com,b.com``)."""
        if isinstance(v, str):
            return tuple(host.strip() for host in v.split(",") if host.strip())
        return v

    log_level: str = Field(default="INFO", description="structlog filter level")
    log_json: bool = Field(default=False, description="render structlog as JSON")

    otel_enabled: bool = Field(default=False, description="ship OTLP spans")
    otel_endpoint: str | None = Field(
        default=None, description="OTLP HTTP endpoint; default uses OTel SDK fallback"
    )

    config_path: str = Field(
        default_factory=lambda: str(magos_home() / "magos.yaml"),
        description=(
            "Path to the routing config YAML. Defaults to $MAGOS_HOME/magos.yaml "
            "(``~/.magos/magos.yaml`` when MAGOS_HOME is unset); override with "
            "MAGOS_CONFIG_PATH or the --config CLI flag. The file must exist; "
            "ship a copy of magos.example.yaml as a starting point."
        ),
    )

    models_path: str | None = Field(
        default=None,
        description=(
            "Override for the registry's models.json location. When set, wins "
            "over yaml's ``registry.models_path``; when unset, the yaml value "
            "(or the derived default ``$MAGOS_HOME/models.json``) applies. "
            "Same path semantics as the yaml field: ``~`` expands against the "
            "OS user home, absolute paths pass through, relative paths anchor "
            "to ``$MAGOS_HOME``."
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
    return MagosSettings()
