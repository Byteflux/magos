"""OpenAI Responses wire shape: ``/v1/responses`` and ``/v1/responses/{id}``."""

from __future__ import annotations

from ._base import Shape, StreamEvent

SPEC = Shape(
    name="openai-responses",
    endpoints=("/v1/responses", "/v1/responses/{id}"),
    compression_provider="openai",
    system_field=None,
    # Responses uses ``input`` (list of items) for conversation; there is
    # no ``messages`` field. Compression rewrites that need a chat-style
    # message list don't apply here.
    messages_field=None,
    instructions_field="instructions",
    usage_keys={
        "input": ("usage", "input_tokens"),
        "output": ("usage", "output_tokens"),
        "cache_read": ("usage", "input_tokens_details", "cached_tokens"),
    },
    stream_events=(
        # Usage arrives on response.completed.response.usage.
        StreamEvent(
            event_name="response.completed",
            usage_path=("response", "usage"),
            model_path=("response", "model"),
            fields={
                "input": ("input_tokens",),
                "output": ("output_tokens",),
                "cache_read": ("input_tokens_details", "cached_tokens"),
            },
        ),
    ),
)
