"""``GET /v1/models`` listing tests.

Covers:

- empty payload when the registry feature is dormant (no refresher)
- OpenAI shape (default) when no Anthropic-flavoured headers are sent
- Anthropic shape selected by ``anthropic-version`` and by ``x-api-key``
- deprecated entries omitted; output sorted by ``namespaced_id``
- ``created`` / ``created_at`` fields derive from
  ``RegistryState.refreshed_at``

The endpoint reads ``app.state.refresher.state`` and nothing else, so
the data-bearing tests assign a tiny stub directly and skip the
lifespan dance; building a real ``Refresher`` would mean fighting
boot discovery and per-provider refresh tasks for no added coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from magos.ingress.http import create_app
from magos.registry.schema import RegistryYaml
from magos.registry.state import ModelEntry, RegistryState
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


@dataclass
class _StubRefresher:
    """Minimal stand-in: the endpoint only ever reads ``.state``."""

    state: RegistryState


def _seeded_state() -> RegistryState:
    """Two live entries across two providers + one soft-deleted entry."""
    live_a = ModelEntry(
        provider="anthropic",
        raw_id="claude-sonnet-4-6",
        litellm_id="anthropic/claude-sonnet-4-6",
        context_size=200_000,
    )
    live_b = ModelEntry(
        provider="openai",
        raw_id="gpt-4o",
        litellm_id="openai/gpt-4o",
        context_size=128_000,
    )
    soft_deleted = ModelEntry(
        provider="openai",
        raw_id="gone",
        litellm_id="openai/gone",
        deprecated_at=datetime(2026, 5, 1, tzinfo=UTC),
    )
    return RegistryState(
        entries={
            live_a.namespaced_id: live_a,
            live_b.namespaced_id: live_b,
            soft_deleted.namespaced_id: soft_deleted,
        },
        refreshed_at={
            "anthropic": datetime(2026, 5, 2, 12, 0, tzinfo=UTC),
            "openai": datetime(2026, 5, 3, 9, 0, tzinfo=UTC),
        },
    )


def _client_with_state(state: RegistryState | None) -> TestClient:
    """Build a client whose ``app.state.refresher`` is a stub or ``None``.

    Skipping the ``with TestClient(...)`` form avoids running the real
    lifespan (which would clobber a stub refresher and try to start
    background tasks).
    """
    app = create_app(routing=_routing_only(), registry=RegistryYaml())
    app.state.refresher = _StubRefresher(state) if state is not None else None
    return TestClient(app)


def test_models_empty_when_no_refresher() -> None:
    client = _client_with_state(None)
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert response.json() == {"object": "list", "data": []}


def test_models_openai_shape_default() -> None:
    client = _client_with_state(_seeded_state())
    response = client.get("/v1/models")
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    ids = [m["id"] for m in payload["data"]]
    # Sorted by namespaced_id; deprecated entry omitted.
    assert ids == ["anthropic/claude-sonnet-4-6", "openai/gpt-4o"]
    by_id = {m["id"]: m for m in payload["data"]}
    assert by_id["anthropic/claude-sonnet-4-6"]["owned_by"] == "anthropic"
    assert by_id["openai/gpt-4o"]["owned_by"] == "openai"
    assert by_id["anthropic/claude-sonnet-4-6"]["created"] == int(
        datetime(2026, 5, 2, 12, 0, tzinfo=UTC).timestamp()
    )
    assert all(m["object"] == "model" for m in payload["data"])


def test_models_anthropic_shape_via_anthropic_version_header() -> None:
    client = _client_with_state(_seeded_state())
    response = client.get("/v1/models", headers={"anthropic-version": "2023-06-01"})
    assert response.status_code == 200
    payload = response.json()
    assert "object" not in payload
    assert payload["has_more"] is False
    assert payload["first_id"] == "anthropic/claude-sonnet-4-6"
    assert payload["last_id"] == "openai/gpt-4o"
    first = payload["data"][0]
    assert first["type"] == "model"
    assert first["id"] == "anthropic/claude-sonnet-4-6"
    assert first["display_name"] == "anthropic/claude-sonnet-4-6"
    assert first["created_at"] == datetime(2026, 5, 2, 12, 0, tzinfo=UTC).isoformat()


def test_models_anthropic_shape_via_x_api_key_header() -> None:
    client = _client_with_state(_seeded_state())
    response = client.get("/v1/models", headers={"x-api-key": "sk-ant-stub"})
    payload = response.json()
    assert "has_more" in payload  # Anthropic shape


def test_models_anthropic_shape_empty_first_last_ids() -> None:
    client = _client_with_state(RegistryState())
    response = client.get("/v1/models", headers={"anthropic-version": "2023-06-01"})
    payload = response.json()
    assert payload == {"data": [], "has_more": False, "first_id": None, "last_id": None}
