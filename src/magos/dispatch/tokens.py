"""Async input-token counting via `litellm.acount_tokens`.

LiteLLM picks between an in-process tokenizer and the provider's native
count-tokens endpoint per model. `count` is a test injection seam.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, Protocol


class CountTokensFn(Protocol):
    def __call__(self, **kwargs: Any) -> Awaitable[Any]: ...


def _extract_total_tokens(result: Any) -> int:
    """Coerce a litellm `TokenCountResponse` or dict into a positive int."""
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
    """Return the input-token count for an Anthropic-shape request."""
    kwargs: dict[str, Any] = {
        "model": dispatch_model,
        "messages": anthropic_request.get("messages") or [],
    }
    for optional in ("system", "tools", "tool_choice"):
        if optional in anthropic_request:
            kwargs[optional] = anthropic_request[optional]
    return _extract_total_tokens(await count(**kwargs))
