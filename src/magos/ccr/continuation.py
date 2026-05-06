"""Build the continuation closure that headroom's CCR handler invokes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Coroutine
from typing import TYPE_CHECKING, Any

from magos.egress.translate.runner import proxy_translate

if TYPE_CHECKING:
    from magos.egress.translate import TranslateAdapter

ContinuationCallable = Callable[
    [list[dict[str, Any]], list[dict[str, Any]] | None],
    Coroutine[Any, Any, dict[str, Any]],
]


def make_continuation_callable(
    *,
    adapter: TranslateAdapter,
    original_body: dict[str, Any],
    completion: Callable[..., Awaitable[Any]],
    dispatch_model: str,
    provider: str,
    forward_headers: dict[str, str],
    api_key: str | None,
    api_base: str | None,
) -> ContinuationCallable:
    """Return a closure suitable as headroom's ``api_call_fn``.

    The closure substitutes ``messages`` and ``tools`` into a copy of
    ``original_body`` and re-runs ``proxy_translate`` with the same egress
    parameters as the caller's original request. Routing engine and
    rewrites are bypassed: the compress rewrite (which produced the CCR
    tool injection in the first place) does NOT run on continuation
    messages, so the freshly-expanded retrieval results are not re-compressed.
    """

    async def _continuation(
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        body = dict(original_body)
        body["messages"] = messages
        if tools is None:
            body.pop("tools", None)
        else:
            body["tools"] = tools
        return await proxy_translate(
            adapter,
            body,
            dispatch_model=dispatch_model,
            provider=provider,
            completion=completion,
            forward_headers=forward_headers,
            api_key=api_key,
            api_base=api_base,
        )

    return _continuation
