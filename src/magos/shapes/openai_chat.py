"""OpenAI Chat Completions wire shape: ``/v1/chat/completions``."""

from __future__ import annotations

from ._spec import ShapeSpec, StreamEvent

SPEC = ShapeSpec(
    name="openai-chat",
    endpoints=("/v1/chat/completions",),
    compression_provider="openai",
    # System prompts ride inside ``messages`` as a role=system entry rather
    # than a top-level field.
    system_field=None,
    messages_field="messages",
    instructions_field=None,
    usage_keys={
        "input": ("usage", "prompt_tokens"),
        "output": ("usage", "completion_tokens"),
        "cache_read": ("usage", "prompt_tokens_details", "cached_tokens"),
    },
    stream_events=(
        # Usage arrives on the terminal chunk regardless of event name,
        # gated by ``stream_options.include_usage: true``. Other chunks
        # have no ``usage`` dict and are skipped by the generic walker.
        StreamEvent(
            event_name=None,
            usage_path=("usage",),
            model_path=("model",),
            fields={
                "input": ("prompt_tokens",),
                "output": ("completion_tokens",),
                "cache_read": ("prompt_tokens_details", "cached_tokens"),
            },
        ),
    ),
)
