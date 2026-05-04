"""Translate-mode dispatch into LiteLLM SDK call sites.

Three endpoint families, each backed by the LiteLLM SDK call that natively
handles its wire shape:

- ``/v1/messages``         -> ``litellm.anthropic_messages`` (Anthropic shape in,
                             Anthropic shape out, including cross-provider routing)
- ``/v1/chat/completions`` -> ``litellm.acompletion``        (OpenAI Chat Completions)
- ``/v1/responses``        -> ``litellm.aresponses``         (OpenAI Responses)

Each entry point accepts a ``completion`` callable for tests; production
wires the matching SDK function. Anthropic streaming returns raw SSE bytes
from LiteLLM, forwarded verbatim. OpenAI streaming wraps chunks into SSE
frames here because the SDK yields parsed objects.

Caller (``magos.egress.dispatch``) supplies ``dispatch_model`` already in
the form LiteLLM expects (``<provider>/<name>`` for unprefixed inputs);
this package no longer infers a provider from the model name.

Per-endpoint logic lives in sibling modules (:mod:`anthropic`,
:mod:`openai_chat`, :mod:`openai_responses`); :mod:`payload` holds the
shared LiteLLM payload builder and process-wide drop_params toggle;
:mod:`sse` has the small SSE framing helpers.
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
