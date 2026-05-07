"""Translate-mode dispatch into LiteLLM SDK call sites.

One endpoint family per sibling module; :mod:`payload` holds the shared
LiteLLM payload builder; :mod:`sse` holds SSE framing helpers. The
generic :mod:`runner` (``proxy_translate`` / ``stream_translate``) is
the entry point for callers — egress dispatch looks up the per-shape
``ADAPTER`` via :data:`TRANSLATE_HANDLERS`. See
``docs/architecture/translation.md``.
"""

from __future__ import annotations

from magos.egress.translate.anthropic import ADAPTER as _ANTHROPIC_ADAPTER
from magos.egress.translate.openai_chat import ADAPTER as _CHAT_ADAPTER
from magos.egress.translate.openai_responses import ADAPTER as _RESPONSES_ADAPTER
from magos.egress.translate.runner import TranslateAdapter

# Endpoint -> adapter lookup used by ``egress.dispatch``.
TRANSLATE_HANDLERS: dict[str, TranslateAdapter] = {
    "/v1/messages": _ANTHROPIC_ADAPTER,
    "/v1/chat/completions": _CHAT_ADAPTER,
    "/v1/responses": _RESPONSES_ADAPTER,
}

__all__ = ["TRANSLATE_HANDLERS", "TranslateAdapter"]
