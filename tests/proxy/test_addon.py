"""Unit tests for the in-process mitmproxy ingress addon.

Two surfaces matter: the ``tls_clienthello`` allowlist gate (off-list
SNI must set ``ignore_connection``), and the ``request`` hook's
host/port/scheme rewrite for allowlisted hosts. Subdomain matching
and the loop-guard (already-target host) are also covered. We use
``mitmproxy.test.tutils.treq`` for real ``Request`` instances and a
lightweight stand-in for ``ClientHelloData`` since constructing a
real one requires a parseable raw ClientHello byte stream.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from mitmproxy.http import HTTPFlow
from mitmproxy.test import tflow, tutils

from magos.proxy.addons.ingress import MagosIngressAddon


def _addon(intercept: tuple[str, ...] = ("api.anthropic.com",)) -> MagosIngressAddon:
    return MagosIngressAddon(
        intercept_hosts=frozenset(intercept),
        target_host="127.0.0.1",
        target_port=6246,
    )


def _flow(host: str, *, port: int = 443, scheme: str = "https") -> HTTPFlow:
    req = tutils.treq(host=host, port=port, scheme=scheme, method="POST", path="/v1/messages")
    return tflow.tflow(req=req)


def _clienthello(sni: str | None) -> SimpleNamespace:
    """Duck-type a ClientHelloData for the SNI hook."""
    return SimpleNamespace(
        client_hello=SimpleNamespace(sni=sni),
        ignore_connection=False,
    )


@pytest.mark.unit
def test_tls_clienthello_passes_through_allowlisted_sni() -> None:
    addon = _addon()
    data = _clienthello("api.anthropic.com")
    addon.tls_clienthello(data)  # type: ignore[arg-type]
    assert data.ignore_connection is False


@pytest.mark.unit
def test_tls_clienthello_passes_through_allowlisted_subdomain() -> None:
    addon = _addon()
    data = _clienthello("eu.api.anthropic.com")
    addon.tls_clienthello(data)  # type: ignore[arg-type]
    assert data.ignore_connection is False


@pytest.mark.unit
def test_tls_clienthello_ignores_off_list_sni() -> None:
    addon = _addon()
    data = _clienthello("example.com")
    addon.tls_clienthello(data)  # type: ignore[arg-type]
    assert data.ignore_connection is True


@pytest.mark.unit
def test_tls_clienthello_ignores_missing_sni() -> None:
    """A CONNECT without SNI (rare but valid TLS) should not be MITM'd."""
    addon = _addon()
    data = _clienthello(None)
    addon.tls_clienthello(data)  # type: ignore[arg-type]
    assert data.ignore_connection is True


@pytest.mark.unit
def test_request_rewrites_allowlisted_host_to_target() -> None:
    addon = _addon()
    flow = _flow("api.anthropic.com")
    addon.request(flow)
    assert flow.request.host == "127.0.0.1"
    assert flow.request.port == 6246
    assert flow.request.scheme == "http"
    # Path/method/body untouched.
    assert flow.request.path == "/v1/messages"
    assert flow.request.method == "POST"


@pytest.mark.unit
def test_request_rewrites_allowlisted_subdomain() -> None:
    addon = _addon()
    flow = _flow("eu.api.anthropic.com")
    addon.request(flow)
    assert flow.request.host == "127.0.0.1"
    assert flow.request.port == 6246


@pytest.mark.unit
def test_request_leaves_off_list_host_alone() -> None:
    addon = _addon()
    flow = _flow("example.com", port=443, scheme="https")
    addon.request(flow)
    assert flow.request.pretty_host == "example.com"
    assert flow.request.port == 443
    assert flow.request.scheme == "https"


@pytest.mark.unit
def test_request_skips_when_already_at_target() -> None:
    """Loop-guard: a re-entrant request from magos's own outbound httpx
    (when ``HTTPS_PROXY`` is set globally) lands here as a request to the
    loopback target. The addon must not re-rewrite; leaving it alone
    keeps the loop visible at most once instead of silently swallowing it.
    """
    addon = _addon(intercept=("127.0.0.1",))  # forced overlap
    flow = _flow("127.0.0.1", port=6246, scheme="http")
    addon.request(flow)
    # Still the same target; addon was a no-op.
    assert flow.request.host == "127.0.0.1"
    assert flow.request.port == 6246
    assert flow.request.scheme == "http"
