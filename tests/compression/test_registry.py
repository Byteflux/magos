"""``PipelineRegistry`` deduplication semantics."""

from __future__ import annotations

from magos.compression import PipelineConfig
from magos.compression.registry import PipelineRegistry, get_registry


def test_same_config_same_provider_returns_same_instance() -> None:
    reg = PipelineRegistry()
    cfg = PipelineConfig()
    a = reg.get_or_build(cfg, provider_name="anthropic")
    b = reg.get_or_build(cfg, provider_name="anthropic")
    assert a is b


def test_different_provider_returns_distinct_instance() -> None:
    reg = PipelineRegistry()
    cfg = PipelineConfig()
    a = reg.get_or_build(cfg, provider_name="anthropic")
    b = reg.get_or_build(cfg, provider_name="openai")
    assert a is not b


def test_different_config_returns_distinct_instance() -> None:
    reg = PipelineRegistry()
    a = reg.get_or_build(PipelineConfig(smart_routing=True), provider_name="anthropic")
    b = reg.get_or_build(PipelineConfig(smart_routing=False), provider_name="anthropic")
    assert a is not b


def test_pipelines_iterates_unique_built_instances() -> None:
    reg = PipelineRegistry()
    cfg = PipelineConfig()
    a = reg.get_or_build(cfg, provider_name="anthropic")
    b = reg.get_or_build(cfg, provider_name="openai")
    seen = list(reg.pipelines())
    assert set(seen) == {a, b}


def test_module_level_registry_is_shared() -> None:
    assert get_registry() is get_registry()
