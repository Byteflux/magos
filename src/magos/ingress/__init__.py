"""In-process mitmproxy ingress proxy.

Run alongside FastAPI in a single process so a client pointed at
``HTTPS_PROXY`` (e.g. Claude Code, which changes behavior when
``ANTHROPIC_BASE_URL`` is set) can reach magos transparently. The
ingress addon terminates TLS for configured hosts and rewrites the
decrypted request to the FastAPI loopback target; everything else
flows through un-MITM'd. See ``docs/ingress.md`` for setup.
"""

from __future__ import annotations

from magos.ingress.addon import MagosIngressAddon
from magos.ingress.log_bridge import StructlogHandler, install_log_bridge
from magos.ingress.master import build_ingress_master

__all__ = [
    "MagosIngressAddon",
    "StructlogHandler",
    "build_ingress_master",
    "install_log_bridge",
]
