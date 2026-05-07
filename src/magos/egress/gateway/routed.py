"""``RoutedGateway``: composite selector picking one Gateway per request."""

from __future__ import annotations

from typing import Any

from fastapi import Response
from fastapi.responses import StreamingResponse

from magos.egress import CompletionFn
from magos.egress.errors import DispatchError
from magos.routing import RouteDecision

from .base import Gateway

_COUNT_TOKENS_ENDPOINT = "/v1/messages/count_tokens"


class RoutedGateway(Gateway):
    """Composite Gateway: picks one of (passthrough / translate / count_tokens).

    Selection rules:

    1. If the request endpoint is ``/v1/messages/count_tokens``, dispatch
       to ``count_tokens`` (regardless of ``target.gateway``).
    2. Otherwise dispatch to ``passthrough`` or ``translate`` per
       ``decision.target.gateway``.
    """

    def __init__(
        self,
        *,
        passthrough: Gateway,
        translate: Gateway,
        count_tokens: Gateway,
    ) -> None:
        self._passthrough = passthrough
        self._translate = translate
        self._count_tokens = count_tokens

    async def dispatch(
        self,
        decision: RouteDecision,
        *,
        completion: CompletionFn,
    ) -> Response | StreamingResponse | dict[str, Any]:
        req = decision.request
        if req.endpoint == _COUNT_TOKENS_ENDPOINT:
            return await self._count_tokens.dispatch(decision, completion=completion)
        gateway_name = decision.target.gateway
        if gateway_name == "passthrough":
            return await self._passthrough.dispatch(decision, completion=completion)
        if gateway_name == "translate":
            return await self._translate.dispatch(decision, completion=completion)
        raise DispatchError(f"unknown gateway: {gateway_name!r}")
