"""Lifespan + admin endpoint smoke tests for the registry-aware server."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from magos.api import build_api
from magos.registry.discovery.base import (
    DiscoveredModel,
    DiscoveryResult,
)
from magos.registry.refresher import Refresher
from magos.registry.schema import ProviderConfig, RegistryYaml
from magos.registry.state import ModelEntry, RegistryState
from magos.registry.store import save as save_state
from magos.routing import RoutingConfig


def _routing_only() -> RoutingConfig:
    return RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "name": "stub",
                    "match": {"model": {"literal": "stub-model"}},
                    "target": {"provider": "x", "gateway": "translate"},
                }
            ]
        }
    )


def _registry_yaml(models_path: Path, providers: dict[str, dict[str, Any]]) -> RegistryYaml:
    return RegistryYaml.model_validate(
        {
            "providers": providers,
            "registry": {"models_path": str(models_path)},
        }
    )


class _StaticAdapter:
    name = "static"

    async def discover(
        self, provider_name: str, config: ProviderConfig, client: httpx.AsyncClient
    ) -> DiscoveryResult:
        return DiscoveryResult(
            models=(
                DiscoveredModel(
                    raw_id="anthropic/claude-sonnet-4-6",
                    litellm_id="openrouter/anthropic/claude-sonnet-4-6",
                ),
            )
        )


def test_app_without_registry_skips_refresher_and_admin(tmp_path: Path) -> None:
    app = build_api(routing=_routing_only(), registry=RegistryYaml())
    assert app.state.refresher is None
    with TestClient(app) as client:
        # /admin/registry not mounted when refresher is None.
        assert client.get("/admin/registry").status_code == 404


def test_app_with_registry_starts_refresher_and_serves_admin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models_path = tmp_path / "models.json"
    # Pre-seed disk so boot discovery is skipped, keeping the test
    # independent of the synthetic adapter being injected.
    entry = ModelEntry(
        provider="openrouter",
        raw_id="anthropic/claude-sonnet-4-6",
        litellm_id="openrouter/anthropic/claude-sonnet-4-6",
        context_size=200000,
    )
    save_state(
        RegistryState(
            entries={entry.namespaced_id: entry},
            refreshed_at={"openrouter": datetime(2026, 5, 2, tzinfo=UTC)},
        ),
        models_path,
    )

    cfg = _registry_yaml(models_path, {"openrouter": {"discovery": "openrouter"}})
    app = build_api(routing=_routing_only(), registry=cfg)
    assert isinstance(app.state.refresher, Refresher)

    with TestClient(app) as client:
        response = client.get("/admin/registry")
        assert response.status_code == 200
        payload = response.json()
        assert any(e["raw_id"] == "anthropic/claude-sonnet-4-6" for e in payload["entries"])


def test_admin_refresh_unknown_provider_returns_404(tmp_path: Path) -> None:
    models_path = tmp_path / "models.json"
    save_state(RegistryState(), models_path)
    cfg = _registry_yaml(models_path, {"openrouter": {"discovery": "openrouter"}})
    app = build_api(routing=_routing_only(), registry=cfg)
    with TestClient(app) as client:
        response = client.post("/admin/registry/refresh", params={"provider": "missing"})
        assert response.status_code == 404


def test_admin_refresh_returns_refreshed_provider_list(tmp_path: Path) -> None:
    """Successful refresh path: noop discovery → refresh() returns clean state."""
    models_path = tmp_path / "models.json"
    save_state(RegistryState(), models_path)
    # `discovery: noop` returns an empty result, so refresh() succeeds
    # without any network. Exercises the success branch of the handler.
    cfg = _registry_yaml(
        models_path,
        {
            "manual": {"discovery": "noop"},
            "other": {"discovery": "noop"},
        },
    )
    app = build_api(routing=_routing_only(), registry=cfg)
    with TestClient(app) as client:
        response = client.post("/admin/registry/refresh")
    assert response.status_code == 200
    payload = response.json()
    assert sorted(payload["refreshed"]) == ["manual", "other"]
    assert payload["failed"] == {}


def test_admin_refresh_scoped_to_one_provider(tmp_path: Path) -> None:
    """`?provider=X` refreshes only that provider."""
    models_path = tmp_path / "models.json"
    save_state(RegistryState(), models_path)
    cfg = _registry_yaml(
        models_path,
        {"manual": {"discovery": "noop"}, "other": {"discovery": "noop"}},
    )
    app = build_api(routing=_routing_only(), registry=cfg)
    with TestClient(app) as client:
        response = client.post("/admin/registry/refresh", params={"provider": "manual"})
    assert response.status_code == 200
    assert response.json()["refreshed"] == ["manual"]


def test_admin_prune_reports_deprecated_counts(tmp_path: Path) -> None:
    """Prune triggers a refresh round and reports deprecated_before/after.

    Pre-seed the registry with one deprecated entry; the noop adapter
    returns an empty discovery result for the provider, so the entry's
    deprecation mark survives the refresh and the before/after counts
    bracket it.
    """
    models_path = tmp_path / "models.json"
    deprecated_entry = ModelEntry(
        provider="manual",
        raw_id="abandoned",
        litellm_id="manual/abandoned",
        deprecated_at=datetime(2026, 5, 2, tzinfo=UTC),
    )
    save_state(
        RegistryState(entries={deprecated_entry.namespaced_id: deprecated_entry}),
        models_path,
    )
    cfg = _registry_yaml(models_path, {"manual": {"discovery": "noop"}})
    app = build_api(routing=_routing_only(), registry=cfg)
    with TestClient(app) as client:
        response = client.post("/admin/registry/prune")
    assert response.status_code == 200
    payload = response.json()
    assert "deprecated_before" in payload
    assert "deprecated_after" in payload
