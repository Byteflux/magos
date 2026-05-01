"""Anthropic <-> OpenAI shape translation.

Covers the static (non-streaming) request and response surfaces of both APIs:
text, images, tool definitions, tool_use / tool_calls, tool_result / tool role
messages, system prompts, stop sequences, and common sampling params. Also
exposes ``AnthropicStreamTranslator`` for converting OpenAI streaming chunks
into Anthropic Messages SSE events.

Public API:

- ``request_anthropic_to_openai``   Anthropic Messages request  -> OpenAI Chat req
- ``response_openai_to_anthropic``  OpenAI Chat response        -> Anthropic Messages resp
- ``request_openai_to_anthropic``   OpenAI Chat request         -> Anthropic Messages request
- ``response_anthropic_to_openai``  Anthropic Messages response -> OpenAI Chat response
- ``AnthropicStreamTranslator``     OpenAI stream chunks        -> Anthropic SSE events

Pydantic models validate at the boundary; output is plain dicts. Models use
``extra="ignore"`` to keep the goldens deterministic. Unknown client fields are
silently dropped, which is acceptable for the proxy until a passthrough policy
is added.
"""

from magos.translation.forward import (
    request_anthropic_to_openai,
    response_openai_to_anthropic,
)
from magos.translation.reverse import (
    request_openai_to_anthropic,
    response_anthropic_to_openai,
)
from magos.translation.streaming import AnthropicStreamTranslator

__all__ = [
    "AnthropicStreamTranslator",
    "request_anthropic_to_openai",
    "request_openai_to_anthropic",
    "response_anthropic_to_openai",
    "response_openai_to_anthropic",
]
