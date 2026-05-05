"""Translate-mode dispatch into LiteLLM SDK call sites.

One endpoint family per sibling module; :mod:`payload` holds the shared
LiteLLM payload builder; :mod:`sse` holds SSE framing helpers. See
``docs/architecture/translation.md``.
"""

from __future__ import annotations

from magos.egress.translate.anthropic import (
    proxy_anthropic_messages,
    stream_anthropic_messages,
)
from magos.egress.translate.openai_chat import (
    proxy_openai_chat_completions,
    stream_openai_chat_completions,
)
from magos.egress.translate.openai_responses import (
    proxy_openai_responses,
    stream_openai_responses,
)

__all__ = [
    "proxy_anthropic_messages",
    "proxy_openai_chat_completions",
    "proxy_openai_responses",
    "stream_anthropic_messages",
    "stream_openai_chat_completions",
    "stream_openai_responses",
]
