"""Anthropic Messages wire shape: ``/v1/messages``."""

from __future__ import annotations

from .base import Shape, StreamEvent

SPEC = Shape(
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
    stream_events=(
        # Input + cache arrive on message_start.message.usage.
        StreamEvent(
            event_name="message_start",
            usage_path=("message", "usage"),
            model_path=("message", "model"),
            fields={
                "input": ("input_tokens",),
                "cache_read": ("cache_read_input_tokens",),
                "cache_write": ("cache_creation_input_tokens",),
            },
        ),
        # Final output arrives on message_delta.usage.
        StreamEvent(
            event_name="message_delta",
            usage_path=("usage",),
            model_path=None,
            fields={"output": ("output_tokens",)},
        ),
    ),
)
