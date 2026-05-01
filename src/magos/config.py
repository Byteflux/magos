"""Declarative configuration for magos.

All runtime knobs live in ``MagosSettings``, populated from environment
variables (prefix ``MAGOS_``) and an optional ``.env`` file. Pydantic enforces
types and ranges at startup so a bad config fails fast instead of surfacing
as a confusing runtime error later.

Example::

    MAGOS_PORT=9000 MAGOS_LOG_JSON=1 python -m magos
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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


def get_settings() -> MagosSettings:
    """Construct fresh settings from the current environment."""
    return MagosSettings()
