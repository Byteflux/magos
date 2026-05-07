"""``Gateway``: ABC for the egress branch.

Implementations encapsulate one external system each:

- :class:`PassthroughGateway` — byte-exact httpx forward
- :class:`TranslateGateway` — LiteLLM SDK call + CCR wrap
- :class:`CountTokensGateway` — LiteLLM count-tokens
- :class:`RoutedGateway` — composite selector that picks one of the above

The selector dispatches on ``decision.request.endpoint`` (count-tokens
endpoint takes its own gateway) and ``decision.target.gateway``
(``"passthrough"`` vs ``"translate"``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from fastapi import Response
from fastapi.responses import StreamingResponse

from magos.dispatch import CompletionFn
from magos.routing import RouteDecision
from magos.routing.request import PostResponseHook
from magos.shapes import Usage
from magos.telemetry import get_logger

log = get_logger("magos.dispatch.gateway")


class Gateway(ABC):
    """Send a routed request to one external system; return its response."""

    @abstractmethod
    async def dispatch(
        self,
        decision: RouteDecision,
        *,
        completion: CompletionFn,
    ) -> Response | StreamingResponse | dict[str, Any]:
        """Dispatch ``decision`` to the underlying system."""


def make_on_complete(
    hooks: list[PostResponseHook],
) -> Callable[[Usage], None] | None:
    """Wrap a hook list into a single on_complete callback.

    Returns None when there are no hooks (so call sites can pass it
    through unconditionally without paying the wrap cost). Each hook
    is fired in order; one raising hook is logged and skipped, the
    rest still fire.
    """
    if not hooks:
        return None
    snapshot = list(hooks)

    def fire(usage: Usage) -> None:
        for hook in snapshot:
            try:
                hook(usage)
            except Exception as exc:
                log.warning(
                    "compress.hook_failed",
                    hook=getattr(hook, "__qualname__", repr(hook)),
                    error=str(exc),
                    error_type=type(exc).__name__,
                )

    return fire
