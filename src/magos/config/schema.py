"""Pydantic schemas for the ``server:`` block in ``magos.yaml``.

Holds two concerns: the FastAPI bind address and the optional in-process
mitmproxy ingress proxy. Both default to off-the-shelf safe values so a
yaml without a ``server:`` block parses cleanly and behaves like before.

``MAGOS_HOST`` / ``MAGOS_PORT`` env vars (via :class:`MagosSettings`)
override ``server.host`` / ``server.port`` at runtime; this module only
declares the yaml shape.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class IngressConfig(_Frozen):
    """In-process mitmproxy ingress proxy configuration.

    When ``enabled`` is true, ``magos serve`` starts a ``DumpMaster`` task
    alongside uvicorn. The addon terminates TLS for hosts in
    ``intercept_hosts`` (and their subdomains) and rewrites the
    decrypted request to the FastAPI loopback target. Requests for hosts
    not on the allowlist flow through un-MITM'd via mitmproxy's
    ``ignore_connection`` mechanism, so an ``HTTPS_PROXY`` pointed at us
    only intercepts what's declared.

    See ``docs/ingress.md`` for the operator-facing setup (CA trust,
    ``HTTPS_PROXY`` configuration, loop-hazard caveat).
    """

    enabled: bool = False
    listen_host: str = Field(default="127.0.0.1", min_length=1)
    listen_port: int = Field(default=8080, ge=1, le=65535)
    intercept_hosts: tuple[str, ...] = ()


class MagosServerConfig(_Frozen):
    """Top-level ``server:`` block: FastAPI bind + ingress."""

    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=8000, ge=1, le=65535)
    ingress: IngressConfig = Field(default_factory=IngressConfig)
