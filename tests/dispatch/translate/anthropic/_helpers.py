"""Shared call-site shorthand for the `/v1/messages` translate tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from magos.dispatch.translate import TRANSLATE_HANDLERS
from magos.dispatch.translate.runner import proxy_translate, stream_translate

ADAPTER = TRANSLATE_HANDLERS["/v1/messages"]


def proxy_anthropic_messages(body: dict[str, Any], **kwargs: Any) -> Any:
    return proxy_translate(ADAPTER, body, **kwargs)


def stream_anthropic_messages(body: dict[str, Any], **kwargs: Any) -> AsyncIterator[bytes]:
    return stream_translate(ADAPTER, body, **kwargs)
