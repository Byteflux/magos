"""Tests for `magos.cli.admin_client`: the httpx wrapper for /admin/registry.

Uses `httpx.MockTransport` to intercept outbound requests so we can
assert URL + method without a running server. Read calls return
`None` on connect failure (CLI falls back to disk); write calls
raise `AdminClientError` so the operator sees the unreachable case.
"""

from __future__ import annotations

import httpx
import pytest

from magos.cli.admin_client import AdminClient, AdminClientError


def _install_transport(
    monkeypatch: pytest.MonkeyPatch, transport: httpx.MockTransport
) -> AdminClient:
    """Build an AdminClient and route its httpx calls through `transport`.

    The class calls module-level `httpx.get` / `httpx.post` directly,
    so we substitute thin wrappers on the imported `httpx` module that
    delegate to a sync `httpx.Client` backed by the mock transport.
    """
    base_url = "http://localhost:6246"
    inner = httpx.Client(transport=transport, base_url=base_url)
    monkeypatch.setattr(httpx, "get", lambda url, **_kw: inner.get(url))
    monkeypatch.setattr(
        httpx,
        "post",
        lambda url, *, params=None, **_kw: inner.post(url, params=params or {}),
    )
    monkeypatch.setattr(
        httpx,
        "request",
        lambda method, url, *, params=None, **_kw: inner.request(method, url, params=params or {}),
    )
    return AdminClient(base_url)


# --- get_registry -----------------------------------------------------


def test_get_registry_returns_response_bytes_on_2xx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b'{"entries": {}}'

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/admin/registry"
        return httpx.Response(200, content=payload)

    client = _install_transport(monkeypatch, httpx.MockTransport(handler))
    assert client.get_registry() == payload


def test_get_registry_returns_none_on_connect_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("server down", request=request)

    client = _install_transport(monkeypatch, httpx.MockTransport(handler))
    assert client.get_registry() is None


def test_get_registry_raises_on_5xx(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"internal error")

    client = _install_transport(monkeypatch, httpx.MockTransport(handler))
    with pytest.raises(AdminClientError, match="500"):
        client.get_registry()


# --- post_refresh -----------------------------------------------------


def test_post_refresh_returns_parsed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_url: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_url["url"] = str(request.url)
        return httpx.Response(200, json={"refreshed": ["openai"], "failed": {}})

    client = _install_transport(monkeypatch, httpx.MockTransport(handler))
    result = client.post_refresh(provider="openai")
    assert result == {"refreshed": ["openai"], "failed": {}}
    assert "provider=openai" in captured_url["url"]


def test_post_refresh_omits_provider_query_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_url: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_url["url"] = str(request.url)
        return httpx.Response(200, json={"refreshed": [], "failed": {}})

    client = _install_transport(monkeypatch, httpx.MockTransport(handler))
    client.post_refresh()
    assert "provider=" not in captured_url["url"]


def test_post_refresh_raises_admin_error_on_connect_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("server down", request=request)

    client = _install_transport(monkeypatch, httpx.MockTransport(handler))
    with pytest.raises(AdminClientError, match="unreachable"):
        client.post_refresh()


def test_post_refresh_raises_admin_error_on_5xx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"upstream timeout")

    client = _install_transport(monkeypatch, httpx.MockTransport(handler))
    with pytest.raises(AdminClientError, match="503"):
        client.post_refresh()


# --- post_prune -------------------------------------------------------


def test_post_prune_returns_parsed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/admin/registry/prune"
        return httpx.Response(200, json={"deprecated_before": 2, "deprecated_after": 0})

    client = _install_transport(monkeypatch, httpx.MockTransport(handler))
    assert client.post_prune() == {"deprecated_before": 2, "deprecated_after": 0}


def test_post_prune_raises_admin_error_on_connect_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("server down", request=request)

    client = _install_transport(monkeypatch, httpx.MockTransport(handler))
    with pytest.raises(AdminClientError, match="unreachable"):
        client.post_prune()


def test_post_prune_raises_admin_error_on_5xx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"boom")

    client = _install_transport(monkeypatch, httpx.MockTransport(handler))
    with pytest.raises(AdminClientError, match="500"):
        client.post_prune()
