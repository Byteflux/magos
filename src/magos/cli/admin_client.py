"""Tiny HTTP client for `/admin/registry`. Reads return `None` if unreachable for disk fallback."""

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

    def _url(self, path: str) -> str:
        return f"{self._base_url}/{path.lstrip('/')}"

    def _check_status(self, response: httpx.Response) -> None:
        if response.is_error:
            raise AdminClientError(f"server returned {response.status_code}: {response.text[:200]}")

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Issue an HTTP request to /admin/registry/<path> and return parsed JSON.

        Raises `AdminClientError` when the server is unreachable or returns an
        error status.
        """
        try:
            response = httpx.request(
                method, self._url(path), params=params or {}, timeout=self._timeout
            )
        except httpx.ConnectError as exc:
            raise AdminClientError(f"server unreachable at {self._base_url}: {exc}") from exc
        self._check_status(response)
        result: dict[str, Any] = response.json()
        return result

    def get_registry(self) -> bytes | None:
        """Raw JSON bytes from `GET /admin/registry`; `None` when unreachable.

        Diverges from `_request` in two ways: the disk-fallback caller
        wants `None` rather than an exception when the server isn't
        running, and the response is forwarded as raw bytes rather than
        re-serialised.
        """
        try:
            response = httpx.get(self._url("/admin/registry"), timeout=self._timeout)
        except httpx.ConnectError:
            return None
        self._check_status(response)
        return response.content

    def post_refresh(self, *, provider: str | None = None) -> dict[str, Any]:
        params = {"provider": provider} if provider else None
        return self._request("POST", "/admin/registry/refresh", params=params)

    def post_prune(self) -> dict[str, Any]:
        return self._request("POST", "/admin/registry/prune")
