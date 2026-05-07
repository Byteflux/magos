"""``TranslateGateway``: LiteLLM SDK marshalling + CCR wrap.

CCR response/stream wrapping lives here; Phase F will optionally extract
it into a ``CCRGateway`` decorator.
"""

from __future__ import annotations

from typing import Any

from fastapi import Response
from fastapi.responses import StreamingResponse

from magos.compression.ccr import wrap_response, wrap_stream
from magos.dispatch import CompletionFn
from magos.dispatch.auth import maybe_inject_api_key, resolve_api_key
from magos.dispatch.errors import DispatchError
from magos.dispatch.gateway.base import Gateway, make_on_complete
from magos.dispatch.translate import TRANSLATE_HANDLERS
from magos.dispatch.translate.runner import proxy_translate, stream_translate
from magos.routing import RouteDecision


class TranslateGateway(Gateway):
    """Translate the request via the per-shape adapter; forward to LiteLLM."""

    async def dispatch(
        self,
        decision: RouteDecision,
        *,
        completion: CompletionFn,
    ) -> Response | StreamingResponse | dict[str, Any]:
        req = decision.request
        target = decision.target

        # Only POST endpoints have litellm equivalents.
        if req.method != "POST":
            raise DispatchError(
                f"gateway='translate' does not support method={req.method!r}; "
                "use gateway='passthrough' for auxiliary GET/DELETE endpoints"
            )

        forward_headers = maybe_inject_api_key(dict(req.headers), target)
        is_streaming = bool(req.body.get("stream"))
        on_complete = make_on_complete(req.post_response_hooks)
        api_key = resolve_api_key(target.api_key_env)
        api_base = target.base_url

        adapter = TRANSLATE_HANDLERS.get(req.endpoint)
        if adapter is None:
            raise DispatchError(f"no translate handler for endpoint {req.endpoint!r}")

        # Shared dispatch parameters; ``proxy_translate``/``stream_translate``
        # also need ``on_complete``, ``wrap_response``/``wrap_stream`` also need
        # ``req`` and ``adapter``. Building from one base dict keeps the two
        # call sites in lock-step when a parameter is added.
        shared: dict[str, Any] = {
            "dispatch_model": decision.dispatch_model,
            "provider": target.provider,
            "completion": completion,
            "forward_headers": forward_headers,
            "api_key": api_key,
            "api_base": api_base,
        }
        translate_kwargs = {**shared, "on_complete": on_complete}
        ccr_kwargs = {**shared, "req": req, "adapter": adapter}

        if is_streaming:
            stream = stream_translate(adapter, dict(req.body), **translate_kwargs)
            return StreamingResponse(
                wrap_stream(stream, **ccr_kwargs),
                media_type="text/event-stream",
            )
        response = await proxy_translate(adapter, dict(req.body), **translate_kwargs)
        return await wrap_response(response, **ccr_kwargs)
