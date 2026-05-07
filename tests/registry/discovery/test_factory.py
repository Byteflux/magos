"""`adapter_for` resolves each known `discovery:` value to its class."""

from __future__ import annotations

from magos.registry.discovery.anthropic import AnthropicAdapter
from magos.registry.discovery.factory import adapter_for
from magos.registry.discovery.noop import NoopAdapter
from magos.registry.discovery.openai import OpenAIAdapter
from magos.registry.discovery.openrouter import OpenRouterAdapter
from magos.registry.discovery.vultr import VultrAdapter
from magos.registry.schema import ProviderConfig


def test_adapter_for_resolves_each_known_kind() -> None:
    assert isinstance(
        adapter_for(ProviderConfig.model_validate({"discovery": "openai"})),
        OpenAIAdapter,
    )
    assert isinstance(
        adapter_for(ProviderConfig.model_validate({"discovery": "anthropic"})),
        AnthropicAdapter,
    )
    assert isinstance(
        adapter_for(ProviderConfig.model_validate({"discovery": "openrouter"})),
        OpenRouterAdapter,
    )
    assert isinstance(
        adapter_for(ProviderConfig.model_validate({"discovery": "vultr"})),
        VultrAdapter,
    )
    assert isinstance(adapter_for(ProviderConfig.model_validate({})), NoopAdapter)
