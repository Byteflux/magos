"""Tiny HTTP client for ``/admin/registry``. Reads return ``None`` if unreachable for disk fallback."""

from __future__ import annotations

from typing import Any

import httpx


class AdminClientError(RuntimeError):
    """Raised when an admin request reaches the server but fails."""


class AdminClient:
    """Synchronous wrapper around the registry admin endpoints."""

    def __init__(self, base_url: str, *, timeout: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Issue an HTTP request to /admin/registry/<path> and return parsed JSON.

        Raises ``AdminClientError`` when the server is unreachable or returns an
        error status.
        """
        url = f"{self._base_url}/{path.lstrip('/')}"
        try:
            response = httpx.request(method, url, params=params or {}, timeout=self._timeout)
        except httpx.ConnectError as exc:
            raise AdminClientError(f"server unreachable at {self._base_url}: {exc}") from exc
        if response.is_error:
            raise AdminClientError(f"server returned {response.status_code}: {response.text[:200]}")
        result: dict[str, Any] = response.json()
        return result

    def get_registry(self) -> bytes | None:
        """Raw JSON bytes from ``GET /admin/registry``; ``None`` when unreachable."""
        try:
            response = httpx.get(f"{self._base_url}/admin/registry", timeout=self._timeout)
        except httpx.ConnectError:
            return None
        if response.is_error:
            raise AdminClientError(f"server returned {response.status_code}: {response.text[:200]}")
        return response.content

    def post_refresh(self, *, provider: str | None = None) -> dict[str, Any]:
        params = {"provider": provider} if provider else None
        return self._request("POST", "/admin/registry/refresh", params=params)

    def post_prune(self) -> dict[str, Any]:
        return self._request("POST", "/admin/registry/prune")
