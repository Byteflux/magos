"""Tests for registry followups: adapter inference + path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from magos.config_loader import resolve_models_path
from magos.registry.discovery.anthropic import AnthropicAdapter
from magos.registry.discovery.factory import adapter_for
from magos.registry.discovery.noop import NoopAdapter
from magos.registry.discovery.openai import OpenAIAdapter
from magos.registry.discovery.openrouter import OpenRouterAdapter
from magos.registry.discovery.vultr import VultrAdapter
from magos.registry.schema import ProviderConfig, RegistryYaml


@pytest.mark.parametrize(
    ("base_url", "expected_cls"),
    [
        ("https://openrouter.ai/api", OpenRouterAdapter),
        ("https://api.anthropic.com", AnthropicAdapter),
        ("https://api.vultrinference.com/v1", VultrAdapter),
        ("https://api.openai.com/v1", OpenAIAdapter),
        ("http://localhost:8001", OpenAIAdapter),
        ("https://generativelanguage.googleapis.com", OpenAIAdapter),
    ],
)
def test_adapter_inferred_from_base_url_when_discovery_unset(
    base_url: str, expected_cls: type
) -> None:
    cfg = ProviderConfig.model_validate({"base_url": base_url})
    assert isinstance(adapter_for(cfg), expected_cls)


def test_adapter_falls_back_to_noop_when_no_base_url() -> None:
    assert isinstance(adapter_for(ProviderConfig.model_validate({})), NoopAdapter)


def test_explicit_discovery_overrides_inference() -> None:
    cfg = ProviderConfig.model_validate(
        {"base_url": "https://openrouter.ai/api", "discovery": "noop"}
    )
    assert isinstance(adapter_for(cfg), NoopAdapter)


def test_resolve_models_path_anchors_relative_to_config_parent(tmp_path: Path) -> None:
    config_path = tmp_path / "config" / "magos.yaml"
    config_path.parent.mkdir()
    config_path.write_text("rules: []", encoding="utf-8")  # content unused
    registry = RegistryYaml.model_validate({"registry": {"models_path": "models.json"}})

    resolved = resolve_models_path(config_path, registry)
    assert resolved == config_path.resolve().parent / "models.json"


def test_resolve_models_path_passes_absolute_through(tmp_path: Path) -> None:
    abs_path = tmp_path / "explicit" / "models.json"
    registry = RegistryYaml.model_validate({"registry": {"models_path": str(abs_path)}})
    assert resolve_models_path("/anywhere/magos.yaml", registry) == abs_path
