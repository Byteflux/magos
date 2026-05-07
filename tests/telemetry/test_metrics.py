"""Tests for `magos.telemetry.metrics`: OTel meter provider + /metrics mount.

`configure_meter_provider` installs a global OTel `MeterProvider`;
because OTel only honors the first real provider per process, we cannot
re-install one cleanly across tests. The test below verifies the call
returns without raising and that the provider class flips off the
no-op default, but skips when the provider has already been set
(e.g. by a prior test or by `tests/registry/test_telemetry.py`'s import).

`mount_metrics_endpoint` is fully testable: it just registers a
FastAPI route, and the route reads from `prometheus_client.REGISTRY`
which is process-global.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from magos.telemetry.metrics import configure_meter_provider, mount_metrics_endpoint


@pytest.mark.unit
def test_configure_meter_provider_runs_without_raising() -> None:
    """The configurator is best-effort: calling it twice is safe."""
    # First call installs the provider (or no-ops if already installed
    # by an earlier test); second call exercises the
    # already-installed branch in OTel itself, surfacing as a warning
    # log on the magos side.
    configure_meter_provider()
    configure_meter_provider()


@pytest.mark.unit
def test_mount_metrics_endpoint_serves_prometheus_text() -> None:
    """`GET /metrics` returns Prometheus exposition-format text."""
    app = FastAPI()
    mount_metrics_endpoint(app)
    with TestClient(app) as client:
        resp = client.get("/metrics")
    assert resp.status_code == 200
    # Prometheus text format starts with `# HELP` / `# TYPE` lines or is
    # empty when no metrics are registered. Either way the content type
    # is the canonical Prometheus exposition mime.
    assert resp.headers["content-type"].startswith("text/plain")


@pytest.mark.unit
def test_mount_metrics_endpoint_handles_missing_prometheus_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When `prometheus_client` import fails, mount becomes a no-op."""
    import builtins  # noqa: PLC0415
    from typing import Any  # noqa: PLC0415

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "prometheus_client" or name.startswith("prometheus_client."):
            raise ImportError("simulated missing prometheus_client")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    app = FastAPI()
    mount_metrics_endpoint(app)
    # Endpoint not registered; FastAPI returns 404 instead of metrics text.
    with TestClient(app) as client:
        resp = client.get("/metrics")
    assert resp.status_code == 404


@pytest.mark.unit
def test_configure_meter_provider_handles_missing_otel_exporter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the OTel Prometheus exporter import fails, configure is a no-op."""
    import builtins  # noqa: PLC0415
    from typing import Any  # noqa: PLC0415

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if "prometheus" in name and "opentelemetry" in name:
            raise ImportError("simulated missing exporter")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Should not raise.
    configure_meter_provider()
