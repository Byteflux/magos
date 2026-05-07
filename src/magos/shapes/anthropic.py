"""Anthropic Messages wire shape: ``/v1/messages``."""

from __future__ import annotations

from ._spec import ShapeSpec

SPEC = ShapeSpec(
    name="anthropic",
    endpoints=("/v1/messages",),
    compression_provider="anthropic",
    system_field="system",
    messages_field="messages",
    instructions_field=None,
    usage_keys={
        "input": ("usage", "input_tokens"),
        "output": ("usage", "output_tokens"),
        "cache_read": ("usage", "cache_read_input_tokens"),
        "cache_write": ("usage", "cache_creation_input_tokens"),
    },
)
