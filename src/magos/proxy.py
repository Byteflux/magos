"""Async pipeline: Anthropic Messages request -> OpenAI dispatch -> Anthropic response.

Pure function. The ``completion`` argument is the seam for tests and routing:
production wires ``litellm.acompletion``; tests inject a fake. Anything that
returns a dict-like or pydantic ``model_dump``-able response works.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol

import litellm

from magos.obs import get_logger, traced
from magos.translation import (
    request_anthropic_to_openai,
    response_openai_to_anthropic,
)

log = get_logger("magos.proxy")


class _CompletionFn(Protocol):
    def __call__(self, **kwargs: Any) -> Awaitable[Any]: ...


def _coerce_to_dict(resp: Any) -> dict[str, Any]:
    if hasattr(resp, "model_dump"):
        dumped: dict[str, Any] = resp.model_dump()
        return dumped
    if isinstance(resp, dict):
        return dict(resp)
    raise TypeError(f"completion returned unsupported type: {type(resp).__name__}")


@traced("proxy.anthropic_messages")
async def proxy_anthropic_messages(
    anthropic_request: dict[str, Any],
    *,
    completion: _CompletionFn | None = None,
) -> dict[str, Any]:
    """Round-trip an Anthropic Messages request through an OpenAI-shape upstream."""
    dispatch: Callable[..., Awaitable[Any]] = completion or litellm.acompletion
    openai_request = request_anthropic_to_openai(anthropic_request)
    log.info("dispatch", model=openai_request.get("model"))
    raw_response = await dispatch(**openai_request)
    openai_response = _coerce_to_dict(raw_response)
    return response_openai_to_anthropic(openai_response)
