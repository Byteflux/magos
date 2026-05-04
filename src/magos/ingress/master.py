"""Build the ``DumpMaster`` that runs alongside FastAPI in-process.

mitmproxy's ``DumpMaster`` is the documented embedding entry point: it
owns the proxy event loop, addon registry, and listener lifecycle.
We construct one with terminal logging and flow-dump rendering both
turned off, since :func:`magos.ingress.log_bridge.install_log_bridge`
already routes mitmproxy log records through structlog.
"""

from __future__ import annotations

from mitmproxy.options import Options
from mitmproxy.tools.dump import DumpMaster

from magos.addon import MagosObserverAddon
from magos.ingress.addon import MagosIngressAddon
from magos.server_config import IngressConfig


def build_ingress_master(
    config: IngressConfig,
    *,
    target_host: str,
    target_port: int,
) -> DumpMaster:
    """Construct an embedded ``DumpMaster`` configured for the ingress addon.

    ``target_host`` / ``target_port`` are where intercepted requests are
    rewritten to — i.e. the FastAPI bind address. The egress observer
    addon (``MagosObserverAddon``) is also loaded so the same mitmproxy
    process can log outbound LLM provider traffic when magos's own
    requests transit it (which they don't by default — see
    ``docs/ingress.md`` "Loop hazard").
    """
    options = Options(
        listen_host=config.listen_host,
        listen_port=config.listen_port,
    )
    master = DumpMaster(options, with_termlog=False, with_dumper=False)
    master.addons.add(  # type: ignore[no-untyped-call]
        MagosIngressAddon(
            intercept_hosts=frozenset(config.intercept_hosts),
            target_host=target_host,
            target_port=target_port,
        ),
        MagosObserverAddon(),
    )
    return master
