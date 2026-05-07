"""Dispatch: how requests leave magos.

Three execution paths chosen by `magos.dispatch.gateway.RoutedGateway`:
`PassthroughGateway` (byte-exact httpx), `TranslateGateway`
(LiteLLM SDK + CCR wrap), `CountTokensGateway` (litellm.acount_tokens).
Auth-header injection lives in `magos.dispatch.auth`. See
`docs/architecture/request-flow.md`.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, Protocol


class CompletionFn(Protocol):
    """Async callable matching `litellm.acompletion` / `aresponses` /
    `anthropic_messages`: keyword-only arguments, returns an awaitable
    yielding either a JSON-shaped dict or a streaming object."""

    def __call__(self, **kwargs: Any) -> Awaitable[Any]: ...


__all__ = ["CompletionFn"]
