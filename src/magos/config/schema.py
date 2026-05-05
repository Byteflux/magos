"""Pydantic schemas for the ``ingress:`` block in ``magos.yaml``.

See ``docs/ingress.md`` and ``docs/architecture/env-vars.md``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class HttpIngressConfig(_Frozen):
    """FastAPI bind address for the primary HTTP ingress."""

    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=6246, ge=1, le=65535)


class MitmIngressConfig(_Frozen):
    """In-process mitmproxy ingress proxy configuration."""

    enabled: bool = False
    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=6247, ge=1, le=65535)
    intercept_hosts: tuple[str, ...] = ()


class MagosIngressConfig(_Frozen):
    """Top-level ``ingress:`` block: HTTP bind + optional mitm proxy."""

    http: HttpIngressConfig = Field(default_factory=HttpIngressConfig)
    mitm: MitmIngressConfig = Field(default_factory=MitmIngressConfig)
