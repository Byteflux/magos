"""Tests for combined config loader: routing + registry from one YAML."""

from __future__ import annotations

from pathlib import Path

import pytest

from magos.config_loader import MagosConfig, load_full_config
from magos.routing.loader import RoutingConfigError


def _write_yaml(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_both_routing_and_registry_blocks(tmp_path: Path) -> None:
    cfg_path = _write_yaml(
        tmp_path / "magos.yaml",
        """
rules:
  - name: pin-haiku
    match:
      model:
        literal: claude-haiku-4-5
    action:
      provider: anthropic
      mode: translate

provider_order:
  - openrouter
  - anthropic

providers:
  openrouter:
    api_key_env: OPENROUTER_API_KEY
    discovery: openrouter
  anthropic:
    api_key_env: ANTHROPIC_API_KEY
    discovery: anthropic_models

registry:
  refresh_interval: 1h
  on_unknown_model: passthrough
""",
    )
    cfg = load_full_config(cfg_path)
    assert isinstance(cfg, MagosConfig)
    assert cfg.routing.rules[0].name == "pin-haiku"
    assert cfg.registry.provider_order == ("openrouter", "anthropic")
    assert cfg.registry.registry.on_unknown_model == "passthrough"
    assert set(cfg.registry.providers) == {"openrouter", "anthropic"}


def test_registry_block_optional(tmp_path: Path) -> None:
    cfg_path = _write_yaml(
        tmp_path / "magos.yaml",
        """
rules:
  - name: only-rule
    match:
      model:
        literal: x
    action:
      provider: openai
      mode: translate
""",
    )
    cfg = load_full_config(cfg_path)
    assert cfg.registry.providers == {}
    assert cfg.registry.provider_order == ()


def test_invalid_registry_block_raises_routing_config_error(tmp_path: Path) -> None:
    cfg_path = _write_yaml(
        tmp_path / "magos.yaml",
        """
rules:
  - name: only-rule
    match:
      model:
        literal: x
    action:
      provider: openai
      mode: translate

registry:
  refresh_interval: not-a-duration
""",
    )
    with pytest.raises(RoutingConfigError, match="invalid registry config"):
        load_full_config(cfg_path)
