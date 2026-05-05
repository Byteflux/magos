"""mitmproxy addon: terminate TLS for allowlisted hosts, rewrite to FastAPI.

Two hooks:

- ``tls_clienthello`` reads the SNI and sets
  ``data.ignore_connection = True`` for any host outside the allowlist.
  mitmproxy then forwards the original CONNECT verbatim and never sees
  the decrypted bytes, so a client pointed at ``HTTPS_PROXY`` can keep
  using unrelated services without breakage.
- ``request`` (fired only for intercepted, decrypted flows) rewrites
  ``host`` / ``port`` / ``scheme`` so the next hop is magos's local
  FastAPI server. Body, headers, method, and path pass through
  untouched, which is what the existing routing rules and byte-exact
  passthrough invariant rely on.

Subdomain matching mirrors :func:`magos.egress.observer._is_llm_host`: any host
whose suffix is ``.<allowed_host>`` matches.
"""

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
        """Skip TLS interception for hosts not on the allowlist.

        Setting ``ignore_connection`` here makes mitmproxy treat the
        CONNECT as opaque: the original TLS handshake passes through
        to the real upstream and we never see the inner bytes.
        """
        sni = data.client_hello.sni
        if sni is None or not self._is_intercepted(sni):
            data.ignore_connection = True

    def request(self, flow: http.HTTPFlow) -> None:
        """Rewrite an intercepted request to point at FastAPI loopback."""
        original_host = flow.request.pretty_host
        if not self._is_intercepted(original_host):
            return
        # Already at the FastAPI target: likely a re-entrant request from
        # magos's own outbound httpx if HTTPS_PROXY is set globally. Leave
        # it alone so the loop is at least visible (not silently swallowed).
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
