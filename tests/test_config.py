"""Unit tests for MagosSettings.

Verifies defaults, env-var overrides, validation bounds, frozen-immutability,
and the deprecation warning for env vars that moved into magos.yaml.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from pydantic import ValidationError

from magos.config import MagosSettings, get_settings


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    # conftest.py sets MAGOS_CONFIG_PATH so create_app() finds the test
    # fixture; clear it here to verify the field default in isolation.
    monkeypatch.delenv("MAGOS_CONFIG_PATH", raising=False)
    monkeypatch.delenv("MAGOS_HOME", raising=False)
    monkeypatch.delenv("MAGOS_MODELS_PATH", raising=False)
    monkeypatch.delenv("MAGOS_HOST", raising=False)
    monkeypatch.delenv("MAGOS_PORT", raising=False)
    s = MagosSettings(_env_file=None)  # type: ignore[call-arg]
    # ``host``/``port`` default to None on MagosSettings now; the actual
    # bind values come from yaml's server block via ``resolve_bind``.
    assert s.host is None
    assert s.port is None
    assert s.log_level == "INFO"
    assert s.log_json is False
    assert s.otel_enabled is False
    assert s.otel_endpoint is None
    assert s.config_path == str(Path.home() / ".magos" / "magos.yaml")
    assert s.models_path is None


def test_magos_home_relocates_config_path_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("MAGOS_CONFIG_PATH", raising=False)
    monkeypatch.setenv("MAGOS_HOME", str(tmp_path / "srv"))
    s = MagosSettings(_env_file=None)  # type: ignore[call-arg]
    assert s.config_path == str(tmp_path / "srv" / "magos.yaml")


def test_explicit_config_path_wins_over_magos_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MAGOS_HOME", str(tmp_path / "srv"))
    monkeypatch.setenv("MAGOS_CONFIG_PATH", "/etc/magos.yaml")
    s = MagosSettings(_env_file=None)  # type: ignore[call-arg]
    assert s.config_path == "/etc/magos.yaml"


def test_models_path_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAGOS_MODELS_PATH", "/var/lib/magos/models.json")
    s = MagosSettings(_env_file=None)  # type: ignore[call-arg]
    assert s.models_path == "/var/lib/magos/models.json"


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAGOS_HOST", "0.0.0.0")
    monkeypatch.setenv("MAGOS_PORT", "9000")
    monkeypatch.setenv("MAGOS_LOG_JSON", "1")
    monkeypatch.setenv("MAGOS_OTEL_ENABLED", "1")
    monkeypatch.setenv("MAGOS_OTEL_ENDPOINT", "http://collector.local:4318/v1/traces")
    monkeypatch.setenv("MAGOS_CONFIG_PATH", "/etc/magos.yaml")

    s = MagosSettings(_env_file=None)  # type: ignore[call-arg]
    assert s.host == "0.0.0.0"
    assert s.port == 9000
    assert s.log_json is True
    assert s.otel_enabled is True
    assert s.otel_endpoint == "http://collector.local:4318/v1/traces"
    assert s.config_path == "/etc/magos.yaml"


def test_invalid_port_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAGOS_PORT", "70000")
    with pytest.raises(ValidationError):
        MagosSettings(_env_file=None)  # type: ignore[call-arg]


def test_settings_are_frozen() -> None:
    s = MagosSettings(_env_file=None)  # type: ignore[call-arg]
    with pytest.raises((ValidationError, dataclasses.FrozenInstanceError, TypeError)):
        s.port = 1234


def test_unknown_env_vars_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAGOS_NOT_A_REAL_FIELD", "value")
    # Should not raise.
    MagosSettings(_env_file=None)  # type: ignore[call-arg]


def test_get_settings_warns_on_removed_env_vars(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("MAGOS_ANTHROPIC_PASSTHROUGH_ENABLED", "1")
    monkeypatch.setenv("MAGOS_COUNT_TOKENS_PASSTHROUGH_PROVIDERS", "anthropic")
    capfd.readouterr()  # discard prior output
    get_settings()
    out = capfd.readouterr().out
    assert "config.removed_env_var" in out
    assert "MAGOS_ANTHROPIC_PASSTHROUGH_ENABLED" in out
    assert "MAGOS_COUNT_TOKENS_PASSTHROUGH_PROVIDERS" in out
