"""Input-token counting strategies for Anthropic Messages requests.

Two strategies, both consumed by ``magos.routing.dispatch``:

- **Local** (always available, sub-millisecond): translate the Anthropic
  request into OpenAI shape and run ``litellm.token_counter``. Works for
  any model LiteLLM recognises.
- **Passthrough** (per-provider, opt-in via the matched rule's
  ``count_tokens_mode: passthrough``): forward to the upstream's native
  count_tokens endpoint for accurate, provider-billed counts. Currently
  implemented for ``anthropic`` via the Anthropic SDK; new providers can
  register an entry in ``PASSTHROUGH_DISPATCH``.

Routing decides which strategy to use; this module exposes the strategies
and the registry. The previous ``count_input_tokens`` orchestrator and the
``resolve_provider`` lookup table moved to the routing layer when
declarative rules subsumed implicit model-prefix routing.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, Protocol

import anthropic
import litellm

from magos.translation import request_anthropic_to_openai
from magos.translation._models import AnthropicCountTokensRequest


def count_locally(anthropic_request: dict[str, Any]) -> int:
    """Sync, in-process estimate via ``litellm.token_counter``.

    Validates against ``AnthropicCountTokensRequest`` (no ``max_tokens``
    required) before reusing the standard request translator. A synthetic
    ``max_tokens`` is injected solely so ``request_anthropic_to_openai``
    accepts the dict; the value is irrelevant for token counting.
    """
    validated = AnthropicCountTokensRequest.model_validate(anthropic_request).model_dump(
        exclude_none=True
    )
    validated["max_tokens"] = 1
    openai_req = request_anthropic_to_openai(validated)
    return int(
        litellm.token_counter(
            model=openai_req["model"],
            messages=openai_req["messages"],
            tools=openai_req.get("tools"),
        )
    )


async def _anthropic_passthrough(
    anthropic_request: dict[str, Any],
    *,
    forward_headers: dict[str, str] | None = None,
) -> int:
    """Forward to Anthropic's native /v1/messages/count_tokens via the SDK.

    ``forward_headers`` is merged into the call's ``extra_headers`` so the
    upstream sees the client's auth, version pins, and beta flags verbatim,
    which preserves Anthropic's billing shape (e.g. ``anthropic-beta``).
    The Anthropic SDK reads ``ANTHROPIC_API_KEY`` from the environment for
    the underlying request; rule-level ``api_key_env`` overrides are not
    plumbed through this strategy in v1.
    """
    validated = AnthropicCountTokensRequest.model_validate(anthropic_request).model_dump(
        exclude_none=True
    )
    kwargs: dict[str, Any] = {"model": validated["model"], "messages": validated["messages"]}
    for optional in ("system", "tools", "tool_choice"):
        if optional in validated:
            kwargs[optional] = validated[optional]
    if forward_headers:
        # Forward only Anthropic-specific knobs to ``extra_headers``: beta
        # flags and version pins. Auth (``x-api-key`` / ``Authorization``)
        # would duplicate the SDK's own auth from ``ANTHROPIC_API_KEY`` and
        # the upstream rejects it; transport headers (``content-type``,
        # ``accept``, ``user-agent``) clobber what the SDK itself sets.
        allowed = {k: v for k, v in forward_headers.items() if k.lower().startswith("anthropic-")}
        if allowed:
            kwargs["extra_headers"] = allowed
    async with anthropic.AsyncAnthropic() as client:
        result = await client.messages.count_tokens(**kwargs)
    return int(result.input_tokens)


class PassthroughFn(Protocol):
    def __call__(
        self,
        anthropic_request: dict[str, Any],
        *,
        forward_headers: dict[str, str] | None = None,
    ) -> Awaitable[int]: ...


PASSTHROUGH_DISPATCH: dict[str, PassthroughFn] = {
    "anthropic": _anthropic_passthrough,
}
