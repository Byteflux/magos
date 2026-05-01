"""Declarative configuration for magos.

All runtime knobs live in ``MagosSettings``, populated from environment
variables (prefix ``MAGOS_``) and an optional ``.env`` file. Pydantic enforces
types and ranges at startup so a bad config fails fast instead of surfacing
as a confusing runtime error later.

Example::

    MAGOS_PORT=9000 MAGOS_LOG_JSON=1 python -m magos
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


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

    anthropic_upstream_url: str = Field(
        default="https://api.anthropic.com",
        description=(
            "Base URL for the upstream Anthropic API used by same-shape "
            "passthrough mode (Anthropic client + anthropic provider)."
        ),
    )
    anthropic_passthrough_enabled: bool = Field(
        default=True,
        description=(
            "When True, /v1/messages requests whose model resolves to the "
            "anthropic provider are forwarded verbatim to the upstream "
            "(preserving OAuth bearer auth, anthropic-beta flags, and "
            "billing surface). When False, the translate-and-dispatch path "
            "is used unconditionally."
        ),
    )

    count_tokens_passthrough_providers: Annotated[frozenset[str], NoDecode] = Field(
        default=frozenset({"anthropic"}),
        description=(
            "LiteLLM provider names whose native count_tokens endpoint is used by "
            "/v1/messages/count_tokens instead of the local estimator. Empty disables "
            "passthrough entirely. Set via comma-separated env var, e.g. "
            "MAGOS_COUNT_TOKENS_PASSTHROUGH_PROVIDERS=anthropic,openai"
        ),
    )

    @field_validator("count_tokens_passthrough_providers", mode="before")
    @classmethod
    def _parse_providers(cls, v: Any) -> Any:
        """Accept comma-separated env strings as well as JSON / native sets."""
        if isinstance(v, str):
            return frozenset(p.strip() for p in v.split(",") if p.strip())
        return v


def get_settings() -> MagosSettings:
    """Construct fresh settings from the current environment."""
    return MagosSettings()
