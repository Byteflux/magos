"""Tests for ``magos.registry.schema`` pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from magos.registry.schema import (
    ModelOverride,
    ProviderConfig,
    RegistrySettings,
    RegistryYaml,
    _parse_duration,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("30s", 30),
        ("2h", 7200),
        ("5m", 300),
        ("1d", 86400),
        ("100ms", 1),  # rounds up to 1s minimum
        ("1500ms", 2),  # rounds to nearest second
        (45, 45),
    ],
)
def test_parse_duration_accepts_supported_units(value: object, expected: int) -> None:
    assert _parse_duration(value) == expected


@pytest.mark.parametrize("bad", ["1y", "abc", "", "10", "h2"])
def test_parse_duration_rejects_invalid_strings(bad: str) -> None:
    with pytest.raises(ValueError):
        _parse_duration(bad)


def test_parse_duration_rejects_non_string_non_int() -> None:
    with pytest.raises(TypeError):
        _parse_duration(2.5)


def test_registry_settings_defaults() -> None:
    s = RegistrySettings()
    assert s.refresh_interval == 7200
    assert s.on_unknown_model == "error"
    assert s.models_path == "./models.json"
    assert s.deprecation_grace_seconds == 3 * 86400
    assert s.discovery_timeout_seconds == 30
    assert s.discovery_max_attempts == 3


def test_registry_settings_parses_duration_strings() -> None:
    s = RegistrySettings.model_validate({"refresh_interval": "1h"})
    assert s.refresh_interval == 3600


def test_registry_settings_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        RegistrySettings.model_validate({"refresh_interval": "1h", "bogus": True})


def test_provider_config_minimal() -> None:
    cfg = ProviderConfig.model_validate({})
    assert cfg.api_key_env is None
    assert cfg.discovery is None
    assert cfg.refresh_interval is None
    assert cfg.models == {}


def test_provider_config_with_models_override() -> None:
    cfg = ProviderConfig.model_validate(
        {
            "api_key_env": "OPENROUTER_API_KEY",
            "discovery": "openrouter",
            "refresh_interval": "4h",
            "models": {
                "anthropic/claude-sonnet-4-6": {
                    "context_size": 1000000,
                    "litellm_id": "openrouter/anthropic/claude-sonnet-4-6:1m",
                },
            },
        }
    )
    assert cfg.refresh_interval == 4 * 3600
    override = cfg.models["anthropic/claude-sonnet-4-6"]
    assert override.context_size == 1000000
    assert override.litellm_id == "openrouter/anthropic/claude-sonnet-4-6:1m"


def test_provider_config_rejects_invalid_discovery_adapter() -> None:
    with pytest.raises(ValidationError):
        ProviderConfig.model_validate({"discovery": "made_up_adapter"})


def test_model_override_rejects_negative_context_size() -> None:
    with pytest.raises(ValidationError):
        ModelOverride.model_validate({"context_size": 0})


def test_registry_yaml_full_round_trip() -> None:
    yaml_dict = {
        "provider_order": ["openrouter", "anthropic"],
        "providers": {
            "openrouter": {
                "api_key_env": "OPENROUTER_API_KEY",
                "discovery": "openrouter",
            },
            "anthropic": {
                "api_key_env": "ANTHROPIC_API_KEY",
                "discovery": "anthropic",
            },
        },
        "registry": {
            "refresh_interval": "1h",
            "on_unknown_model": "passthrough",
        },
    }
    parsed = RegistryYaml.model_validate(yaml_dict)
    assert parsed.provider_order == ("openrouter", "anthropic")
    assert parsed.registry.refresh_interval == 3600
    assert parsed.registry.on_unknown_model == "passthrough"
    assert set(parsed.providers) == {"openrouter", "anthropic"}


def test_registry_yaml_defaults_when_blocks_omitted() -> None:
    parsed = RegistryYaml.model_validate({})
    assert parsed.providers == {}
    assert parsed.provider_order == ()
    assert parsed.registry.on_unknown_model == "error"
