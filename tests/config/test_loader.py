"""Tests for combined config loader: routing + registry from one YAML."""

from __future__ import annotations

from pathlib import Path

import pytest

from magos.config.loader import MagosConfig, load_full_config
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
    target:
      provider: anthropic
      gateway: translate

provider_order:
  - openrouter
  - anthropic

providers:
  openrouter:
    api_key_env: OPENROUTER_API_KEY
    discovery: openrouter
  anthropic:
    api_key_env: ANTHROPIC_API_KEY
    discovery: anthropic

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
    target:
      provider: openai
      gateway: translate
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
    target:
      provider: openai
      gateway: translate

registry:
  refresh_interval: not-a-duration
""",
    )
    with pytest.raises(RoutingConfigError, match="invalid registry config"):
        load_full_config(cfg_path)


def test_provider_base_url_filled_from_adapter_default(tmp_path: Path) -> None:
    """Operators omitting ``base_url`` get the adapter's canonical URL.

    Vultr is the canary: it ships through ``custom_openai`` and therefore
    needs an explicit api_base at dispatch time, but the URL is well-known
    and shouldn't have to be repeated in every operator's yaml.
    """
    cfg_path = _write_yaml(
        tmp_path / "magos.yaml",
        """
rules:
  - name: only-rule
    match:
      model:
        literal: x
    target:
      provider: openai
      gateway: translate

providers:
  vultr:
    api_key_env: VULTR_API_KEY
    discovery: vultr
  openai:
    api_key_env: OPENAI_API_KEY
    discovery: openai
  anthropic:
    api_key_env: ANTHROPIC_API_KEY
    discovery: anthropic
""",
    )
    cfg = load_full_config(cfg_path)
    assert cfg.registry.providers["vultr"].base_url == "https://api.vultrinference.com/v1"
    assert cfg.registry.providers["openai"].base_url == "https://api.openai.com"
    # Adapters with no canonical URL stay None; LiteLLM's per-provider
    # default handles both discovery and dispatch.
    assert cfg.registry.providers["anthropic"].base_url is None


def test_explicit_provider_base_url_overrides_adapter_default(tmp_path: Path) -> None:
    """Operator-supplied ``base_url`` wins over adapter default."""
    cfg_path = _write_yaml(
        tmp_path / "magos.yaml",
        """
rules:
  - name: only-rule
    match:
      model:
        literal: x
    target:
      provider: openai
      gateway: translate

providers:
  vultr:
    api_key_env: VULTR_API_KEY
    base_url: https://internal-vultr-proxy.example.com/v1
    discovery: vultr
""",
    )
    cfg = load_full_config(cfg_path)
    assert cfg.registry.providers["vultr"].base_url == "https://internal-vultr-proxy.example.com/v1"
