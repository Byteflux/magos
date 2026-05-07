"""In-process mitmproxy ingress: terminates TLS for allowlisted hosts
and rewrites to the FastAPI loopback target so ``HTTPS_PROXY`` clients
(e.g. Claude Code) reach magos transparently. See ``docs/ingress.md``."""

from __future__ import annotations

from magos.proxy.addons.ingress import MagosIngressAddon
from magos.proxy.listener import build_ingress_master
from magos.proxy.log_bridge import StructlogHandler, install_log_bridge

__all__ = [
    "MagosIngressAddon",
    "StructlogHandler",
    "build_ingress_master",
    "install_log_bridge",
]
