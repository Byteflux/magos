"""In-process mitmproxy ingress: terminates TLS for allowlisted hosts
and rewrites to the FastAPI loopback target so ``HTTPS_PROXY`` clients
(e.g. Claude Code) reach magos transparently. See ``docs/ingress.md``."""

from __future__ import annotations

from magos.ingress.mitm.addon import MagosIngressAddon
from magos.ingress.mitm.log_bridge import StructlogHandler, install_log_bridge
from magos.ingress.mitm.master import build_ingress_master

__all__ = [
    "MagosIngressAddon",
    "StructlogHandler",
    "build_ingress_master",
    "install_log_bridge",
]
