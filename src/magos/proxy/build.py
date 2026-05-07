"""Composition root for the mitmproxy ``HTTPS_PROXY`` surface. The only
place that imports broadly across magos packages to assemble the object
graph for the embedded proxy process.

Termlog + dumper disabled because
:func:`magos.proxy.log_bridge.install_log_bridge` already routes
mitmproxy logs through structlog.
"""

from __future__ import annotations

from mitmproxy.options import Options
from mitmproxy.tools.dump import DumpMaster

from magos.config.schema import MitmIngressConfig
from magos.proxy.addons.ingress import MagosIngressAddon
from magos.proxy.addons.observer import MagosObserverAddon


def build_proxy(
    config: MitmIngressConfig,
    *,
    target_host: str,
    target_port: int,
) -> DumpMaster:
    """Construct the embedded mitmproxy ``DumpMaster`` with addons wired.

    ``target_host``/``target_port`` is the FastAPI bind address that
    intercepted requests get rewritten to. ``MagosObserverAddon`` is also
    loaded for outbound provider traffic if it transits this proxy
    (it doesn't by default; see ``docs/ingress.md`` "Loop hazard").
    """
    options = Options(
        listen_host=config.host,
        listen_port=config.port,
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
