"""Input-token counting for Anthropic Messages requests.

Two strategies are available:

- **Local** (always available, sub-millisecond): translate the Anthropic
  request into OpenAI shape and run ``litellm.token_counter``. Works for any
  model LiteLLM recognises.
- **Passthrough** (opt-in per provider): forward to the upstream's native
  count_tokens endpoint for accurate, provider-billed counts. Currently
  implemented for ``anthropic`` via the Anthropic SDK; other providers fall
  back to local with a warning.

Streaming uses local-only to avoid adding upstream latency to time-to-first-
byte; the ``/v1/messages/count_tokens`` HTTP endpoint can use passthrough
when the resolved provider is allow-listed in
``MagosSettings.count_tokens_passthrough_providers``.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, Protocol

import anthropic
import litellm

from magos.obs import get_logger
from magos.translation import request_anthropic_to_openai

log = get_logger("magos.tokens")


def count_locally(anthropic_request: dict[str, Any]) -> int:
    """Sync, in-process estimate via ``litellm.token_counter``."""
    openai_req = request_anthropic_to_openai(anthropic_request)
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
    """
    kwargs: dict[str, Any] = {
        "model": anthropic_request["model"],
        "messages": anthropic_request["messages"],
    }
    for optional in ("system", "tools", "tool_choice"):
        value = anthropic_request.get(optional)
        if value is not None:
            kwargs[optional] = value
    if forward_headers:
        kwargs["extra_headers"] = forward_headers
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

# Fallback mapping for unprefixed model names that LiteLLM does not resolve
# on its own (e.g. ``claude-3-5-sonnet-20241022`` vs the prefixed
# ``anthropic/claude-...``). The proper fix is alias-based routing config;
# this keeps token counting useful for clients calling magos with bare names.
_BARE_MODEL_PROVIDER_FALLBACK: tuple[tuple[str, str], ...] = (
    ("claude-", "anthropic"),
    ("gpt-", "openai"),
    ("o1-", "openai"),
    ("o3-", "openai"),
    ("o4-", "openai"),
    ("gemini-", "vertex_ai"),
    ("mistral-", "mistral"),
    ("command-", "cohere"),
)


def resolve_provider(model: str) -> str:
    """Return the LiteLLM provider name for ``model``, or '' if unknown."""
    try:
        _, provider, _, _ = litellm.get_llm_provider(model)
        return str(provider)
    except Exception:  # noqa: S110
        # LiteLLM rejects bare names like ``claude-3-5-sonnet-...`` without a
        # provider prefix; fall through to the prefix table below.
        pass
    lowered = model.lower()
    for prefix, provider in _BARE_MODEL_PROVIDER_FALLBACK:
        if lowered.startswith(prefix):
            return provider
    return ""


# Backwards-compat alias for internal callers; prefer ``resolve_provider``.
_resolve_provider = resolve_provider


async def count_input_tokens(
    anthropic_request: dict[str, Any],
    *,
    passthrough_providers: frozenset[str] = frozenset(),
    forward_headers: dict[str, str] | None = None,
) -> int:
    """Count input tokens for an Anthropic Messages request.

    Uses passthrough when the request's model resolves to a provider in
    ``passthrough_providers`` and a passthrough implementation is registered
    in ``PASSTHROUGH_DISPATCH``. ``forward_headers`` is forwarded to the
    passthrough call so upstream auth, version pins, and beta flags are
    preserved. Falls back to ``count_locally`` on any error so the endpoint
    stays available even if upstream credentials are missing.
    """
    if passthrough_providers:
        provider = resolve_provider(str(anthropic_request.get("model", "")))
        impl = PASSTHROUGH_DISPATCH.get(provider) if provider in passthrough_providers else None
        if impl is not None:
            try:
                return await impl(anthropic_request, forward_headers=forward_headers)
            except Exception as exc:
                log.warning(
                    "count_tokens.passthrough_failed",
                    provider=provider,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
    return count_locally(anthropic_request)
