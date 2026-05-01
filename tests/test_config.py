"""Unit tests for MagosSettings.

Verifies defaults, env-var overrides, validation bounds, and that the
settings object is frozen so callers cannot mutate config at runtime.
"""

from __future__ import annotations

import dataclasses

import pytest
from pydantic import ValidationError

from magos.config import MagosSettings


def test_defaults() -> None:
    s = MagosSettings(_env_file=None)  # type: ignore[call-arg]
    assert s.host == "127.0.0.1"
    assert s.port == 8000
    assert s.log_level == "INFO"
    assert s.log_json is False
    assert s.otel_enabled is False
    assert s.otel_endpoint is None


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAGOS_HOST", "0.0.0.0")
    monkeypatch.setenv("MAGOS_PORT", "9000")
    monkeypatch.setenv("MAGOS_LOG_JSON", "1")
    monkeypatch.setenv("MAGOS_OTEL_ENABLED", "1")
    monkeypatch.setenv("MAGOS_OTEL_ENDPOINT", "http://collector.local:4318/v1/traces")

    s = MagosSettings(_env_file=None)  # type: ignore[call-arg]
    assert s.host == "0.0.0.0"
    assert s.port == 9000
    assert s.log_json is True
    assert s.otel_enabled is True
    assert s.otel_endpoint == "http://collector.local:4318/v1/traces"


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


def test_count_tokens_passthrough_default() -> None:
    s = MagosSettings(_env_file=None)  # type: ignore[call-arg]
    assert s.count_tokens_passthrough_providers == frozenset({"anthropic"})


def test_count_tokens_passthrough_csv_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAGOS_COUNT_TOKENS_PASSTHROUGH_PROVIDERS", "anthropic, openai , vertex_ai")
    s = MagosSettings(_env_file=None)  # type: ignore[call-arg]
    assert s.count_tokens_passthrough_providers == frozenset({"anthropic", "openai", "vertex_ai"})


def test_count_tokens_passthrough_disabled_via_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAGOS_COUNT_TOKENS_PASSTHROUGH_PROVIDERS", "")
    s = MagosSettings(_env_file=None)  # type: ignore[call-arg]
    assert s.count_tokens_passthrough_providers == frozenset()
