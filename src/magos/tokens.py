"""Async input-token counting via ``litellm.acount_tokens``.

LiteLLM picks between an in-process tokenizer and the provider's native
count-tokens endpoint based on the model's provider, so a single call
covers both the local-estimate and provider-billed paths.

The ``count`` argument is the seam for tests: production wires
``litellm.acount_tokens``; tests inject a fake. Anything that returns an
object with a ``total_tokens`` attribute or a dict carrying that key works.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, Protocol


class CountTokensFn(Protocol):
    def __call__(self, **kwargs: Any) -> Awaitable[Any]: ...


def _extract_total_tokens(result: Any) -> int:
    """Coerce a litellm ``TokenCountResponse`` or dict into a positive int."""
    if hasattr(result, "total_tokens"):
        value = result.total_tokens
    elif isinstance(result, dict):
        value = result.get("total_tokens")
    else:
        raise TypeError(f"count_tokens returned unsupported type: {type(result).__name__}")
    if not isinstance(value, int):
        raise TypeError(f"count_tokens returned non-int total_tokens: {value!r}")
    return value


async def count_tokens(
    anthropic_request: dict[str, Any],
    *,
    dispatch_model: str,
    count: CountTokensFn,
) -> int:
    """Return the input-token count for an Anthropic-shape request.

    ``dispatch_model`` is the LiteLLM-prefixed model id chosen by routing
    (e.g. ``anthropic/claude-haiku-4-5``); the inbound body's model may be
    a bare alias the operator declared. ``count`` is the injected SDK call
    (``litellm.acount_tokens`` in production).
    """
    kwargs: dict[str, Any] = {
        "model": dispatch_model,
        "messages": anthropic_request.get("messages") or [],
    }
    for optional in ("system", "tools", "tool_choice"):
        if optional in anthropic_request:
            kwargs[optional] = anthropic_request[optional]
    return _extract_total_tokens(await count(**kwargs))
