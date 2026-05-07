"""`GET /v1/models`: list live registry entries in OpenAI or Anthropic
shape (selected by `anthropic-version` / `x-api-key` header). Empty
list when registry is dormant. See `docs/registry.md`."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from magos.registry.refresher import Refresher
from magos.registry.state import ModelEntry

_EPOCH = datetime.fromtimestamp(0, tz=UTC)


def _is_anthropic_shape(request: Request) -> bool:
    headers = request.headers
    return "anthropic-version" in headers or "x-api-key" in headers


def _live_entries(refresher: Refresher | None) -> list[ModelEntry]:
    if refresher is None:
        return []
    entries = [e for e in refresher.state.entries.values() if not e.is_deprecated]
    entries.sort(key=lambda e: e.namespaced_id)
    return entries


def _refreshed_at(refresher: Refresher | None, provider: str) -> datetime:
    if refresher is None:
        return _EPOCH
    return refresher.state.refreshed_at.get(provider) or _EPOCH


def _openai_payload(entries: list[ModelEntry], refresher: Refresher | None) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": e.namespaced_id,
                "object": "model",
                "created": int(_refreshed_at(refresher, e.provider).timestamp()),
                "owned_by": e.provider,
            }
            for e in entries
        ],
    }


def _anthropic_payload(entries: list[ModelEntry], refresher: Refresher | None) -> dict[str, Any]:
    data = [
        {
            "type": "model",
            "id": e.namespaced_id,
            "display_name": e.namespaced_id,
            "created_at": _refreshed_at(refresher, e.provider).isoformat(),
        }
        for e in entries
    ]
    return {
        "data": data,
        "has_more": False,
        "first_id": data[0]["id"] if data else None,
        "last_id": data[-1]["id"] if data else None,
    }


def register_models_endpoint(app: FastAPI) -> None:

    @app.get("/v1/models")
    async def list_models(request: Request) -> JSONResponse:  # type: ignore[unused-ignore]
        refresher = cast("Refresher | None", request.app.state.refresher)
        entries = _live_entries(refresher)
        if _is_anthropic_shape(request):
            return JSONResponse(_anthropic_payload(entries, refresher))
        return JSONResponse(_openai_payload(entries, refresher))
