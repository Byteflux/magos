"""``CountTokensGateway``: ``litellm.acount_tokens`` for ``/v1/messages/count_tokens``."""

from __future__ import annotations

from typing import Any

from fastapi import Response
from fastapi.responses import StreamingResponse

from magos.egress import CompletionFn
from magos.egress.tokens import count_tokens
from magos.routing import RouteDecision

from .base import Gateway


class CountTokensGateway(Gateway):
    """Return the input-token count for an Anthropic-shape request.

    Selected by the ``RoutedGateway`` when the endpoint is
    ``/v1/messages/count_tokens`` (regardless of ``target.gateway``).
    """

    async def dispatch(
        self,
        decision: RouteDecision,
        *,
        completion: CompletionFn,
    ) -> Response | StreamingResponse | dict[str, Any]:
        req = decision.request
        n = await count_tokens(
            dict(req.body),
            dispatch_model=decision.dispatch_model,
            count=completion,
        )
        return {"input_tokens": n}
