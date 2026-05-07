"""In-process mitmproxy ingress: terminates TLS for allowlisted hosts
and rewrites to the FastAPI loopback target so `HTTPS_PROXY` clients
(e.g. Claude Code) reach magos transparently. See `docs/ingress.md`."""

from __future__ import annotations

from magos.proxy.build import build_proxy

__all__ = ["build_proxy"]
