"""Tests for ``magos models`` CLI subcommands."""

from __future__ import annotations

import io
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from magos.cli import models_cmd
from magos.cli.admin_client import AdminClientError
from magos.config import MagosSettings
from magos.registry.models import ModelEntry, RegistryState
from magos.registry.store import save as save_state
from magos.registry.store import serialize


def _entry() -> ModelEntry:
    return ModelEntry(
        provider="openrouter",
        raw_id="anthropic/claude-sonnet-4-6",
        litellm_id="openrouter/anthropic/claude-sonnet-4-6",
        context_size=200000,
    )


def _state() -> RegistryState:
    e = _entry()
    return RegistryState(
        entries={e.namespaced_id: e},
        refreshed_at={"openrouter": datetime(2026, 5, 2, tzinfo=UTC)},
    )


@pytest.fixture
def fake_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MagosSettings:
    """Point settings at a tmp config that declares only the registry block."""
    cfg_path = tmp_path / "magos.yaml"
    cfg_path.write_text(
        """
rules:
  - name: stub
    match:
      model:
        literal: never-matches
    action:
      provider: x
      mode: translate

registry:
  models_path: """
        + str(tmp_path / "models.json").replace("\\", "/")
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MAGOS_CONFIG_PATH", str(cfg_path))
    return MagosSettings()


def test_list_falls_back_to_disk_when_server_unreachable(
    monkeypatch: pytest.MonkeyPatch, fake_settings: MagosSettings, tmp_path: Path
) -> None:
    save_state(_state(), Path(fake_settings.config_path).parent / "models.json")

    class _UnreachableClient:
        def get_registry(self) -> bytes | None:
            return None

    monkeypatch.setattr(models_cmd, "_admin_client", lambda _s: _UnreachableClient())

    out = io.StringIO()
    rc = models_cmd.main(["list"], out=out)
    assert rc == 0
    assert "openrouter/anthropic/claude-sonnet-4-6" in out.getvalue()
    assert "falling back to disk" in out.getvalue()


def test_list_prefers_server_state(
    monkeypatch: pytest.MonkeyPatch, fake_settings: MagosSettings
) -> None:
    payload = serialize(_state())

    class _ServerClient:
        def get_registry(self) -> bytes | None:
            return payload

    monkeypatch.setattr(models_cmd, "_admin_client", lambda _s: _ServerClient())

    out = io.StringIO()
    rc = models_cmd.main(["list"], out=out)
    assert rc == 0
    assert "# source: server" in out.getvalue()


def test_list_from_disk_skips_server(
    monkeypatch: pytest.MonkeyPatch, fake_settings: MagosSettings, tmp_path: Path
) -> None:
    save_state(_state(), Path(fake_settings.config_path).parent / "models.json")

    called = False

    class _ShouldNotBeCalled:
        def get_registry(self) -> bytes | None:
            nonlocal called
            called = True
            return None

    monkeypatch.setattr(models_cmd, "_admin_client", lambda _s: _ShouldNotBeCalled())
    out = io.StringIO()
    rc = models_cmd.main(["list", "--from-disk"], out=out)
    assert rc == 0
    assert not called
    assert "# source: disk" in out.getvalue()


def test_show_returns_nonzero_on_unknown_id(
    monkeypatch: pytest.MonkeyPatch, fake_settings: MagosSettings, tmp_path: Path
) -> None:
    save_state(RegistryState(), Path(fake_settings.config_path).parent / "models.json")
    monkeypatch.setattr(
        models_cmd,
        "_admin_client",
        lambda _s: type("X", (), {"get_registry": lambda self: None})(),
    )
    out = io.StringIO()
    rc = models_cmd.main(["show", "missing/x"], out=out)
    assert rc == 1
    assert "not found" in out.getvalue()


def test_refresh_returns_server_response(
    monkeypatch: pytest.MonkeyPatch, fake_settings: MagosSettings
) -> None:
    class _OK:
        def post_refresh(self, *, provider: str | None = None) -> dict[str, Any]:
            return {"refreshed": ["openrouter"], "failed": {}}

    monkeypatch.setattr(models_cmd, "_admin_client", lambda _s: _OK())
    out = io.StringIO()
    rc = models_cmd.main(["refresh", "--provider", "openrouter"], out=out)
    assert rc == 0
    assert "refreshed" in out.getvalue()


def test_refresh_returns_nonzero_on_partial_failure(
    monkeypatch: pytest.MonkeyPatch, fake_settings: MagosSettings
) -> None:
    class _PartialFail:
        def post_refresh(self, *, provider: str | None = None) -> dict[str, Any]:
            return {"refreshed": [], "failed": {"openrouter": "boom"}}

    monkeypatch.setattr(models_cmd, "_admin_client", lambda _s: _PartialFail())
    out = io.StringIO()
    rc = models_cmd.main(["refresh"], out=out)
    assert rc == 1


def test_refresh_returns_nonzero_when_server_unreachable(
    monkeypatch: pytest.MonkeyPatch, fake_settings: MagosSettings
) -> None:
    class _Unreachable:
        def post_refresh(self, *, provider: str | None = None) -> dict[str, Any]:
            raise AdminClientError("server unreachable at http://localhost:8000")

    monkeypatch.setattr(models_cmd, "_admin_client", lambda _s: _Unreachable())
    out = io.StringIO()
    rc = models_cmd.main(["refresh"], out=out)
    assert rc == 2
    assert "server unreachable" in out.getvalue()
