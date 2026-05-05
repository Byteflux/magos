"""mitmproxy addon: TLS-terminate allowlisted hosts, rewrite to FastAPI.
Non-allowlisted CONNECTs are passed through opaque (``ignore_connection``).
Subdomain match mirrors :func:`magos.egress.observer._is_llm_host`. See
``docs/ingress.md``."""

from __future__ import annotations

from mitmproxy import http, tls

from magos.telemetry import get_logger

log = get_logger("magos.ingress.mitm.addon")


class MagosIngressAddon:
    """Rewrite HTTPS-proxied requests for allowlisted hosts to FastAPI loopback."""

    def __init__(
        self,
        intercept_hosts: frozenset[str],
        target_host: str,
        target_port: int,
    ) -> None:
        self._hosts = intercept_hosts
        self._target_host = target_host
        self._target_port = target_port

    def tls_clienthello(self, data: tls.ClientHelloData) -> None:
        """Skip TLS interception for non-allowlisted SNIs (sets
        ``ignore_connection`` so the original handshake passes through)."""
        sni = data.client_hello.sni
        if sni is None or not self._is_intercepted(sni):
            data.ignore_connection = True

    def request(self, flow: http.HTTPFlow) -> None:
        """Rewrite an intercepted request to point at FastAPI loopback."""
        original_host = flow.request.pretty_host
        if not self._is_intercepted(original_host):
            return
        # Already at the FastAPI target: likely a re-entrant outbound
        # httpx request if HTTPS_PROXY is set globally. Leave alone so
        # the loop stays visible rather than silently swallowed.
        if flow.request.host == self._target_host and flow.request.port == self._target_port:
            return
        flow.request.host = self._target_host
        flow.request.port = self._target_port
        flow.request.scheme = "http"
        log.info(
            "ingress.rewrote",
            original_host=original_host,
            target=f"{self._target_host}:{self._target_port}",
            method=flow.request.method,
            path=flow.request.path,
        )

    def _is_intercepted(self, host: str) -> bool:
        if host in self._hosts:
            return True
        return any(host.endswith(f".{h}") for h in self._hosts)
