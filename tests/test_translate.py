"""Unit tests for the LiteLLM-backed proxy entry points.

After the LiteLLM SDK fold-in there is no per-field translation to verify;
``proxy_anthropic_messages`` is a thin marshalling layer over
``litellm.anthropic_messages``. These tests check that contract: payload
composition (model rewrite, header forwarding, api_key threading) and
response coercion. The cross-provider behavior itself is covered by the
e2e suite (``MAGOS_E2E=1``).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from magos.egress.translate import proxy_anthropic_messages, stream_anthropic_messages


@pytest.mark.unit
def test_proxy_module_enables_litellm_drop_params() -> None:
    """Importing ``magos.egress.translate`` must flip ``litellm.drop_params`` to True.

    Cross-shape translation (Anthropic <-> OpenAI) routinely sends params one
    side supports and the other does not. ``context_management`` from Claude
    Code on an upstream routed via ``custom_openai`` is the canary: without
    drop_params LiteLLM raises ``UnsupportedParamsError`` and the request
    fails before reaching the provider.
    """
    import litellm  # noqa: PLC0415

    import magos.egress.translate  # noqa: F401, PLC0415

    assert litellm.drop_params is True


@pytest.mark.unit
def test_proxy_anthropic_messages_strips_unknown_fields_for_non_anthropic() -> None:
    """Anthropic-only body fields must not leak through to non-Anthropic upstreams.

    LiteLLM's ``anthropic_messages`` only translates canonical Anthropic
    Messages fields to OpenAI shape; unknown fields fall through to ``**kwargs``
    and surface inside the destination SDK as ``unexpected keyword argument``
    errors (e.g. ``output_config``, ``context_management`` from Claude Code).
    Verify the proxy strips them before dispatch when the target is not
    Anthropic, and leaves them alone when it is.
    """
    seen: dict[str, Any] = {}

    async def fake(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"id": "ok", "content": []}

    body = {
        "model": "claude-x",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
        "context_management": {"edits": []},
        "output_config": {
            "effort": "xhigh",
            "format": {
                "type": "json_schema",
                "schema": {"type": "object", "properties": {"name": {"type": "string"}}},
            },
        },
    }
    asyncio.run(
        proxy_anthropic_messages(
            body,
            dispatch_model="custom_openai/zai-org/GLM-5.1-FP8",
            completion=fake,
        )
    )
    # Anthropic-only fields stripped; output_config translated to OpenAI shape.
    assert "context_management" not in seen
    assert "output_config" not in seen
    assert seen["messages"] == body["messages"]
    assert seen["reasoning_effort"] == "high"  # xhigh clamps to high
    assert seen["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "schema": {"type": "object", "properties": {"name": {"type": "string"}}},
        },
    }

    seen.clear()
    asyncio.run(
        proxy_anthropic_messages(
            body,
            dispatch_model="anthropic/claude-x",
            completion=fake,
        )
    )
    # Anthropic-bound: forwarded verbatim, no translation.
    assert seen.get("context_management") == {"edits": []}
    assert seen.get("output_config") == body["output_config"]
    assert "reasoning_effort" not in seen
    assert "response_format" not in seen


@pytest.mark.unit
def test_proxy_anthropic_messages_passes_dispatch_model_and_returns_dict() -> None:
    received: dict[str, Any] = {}

    async def fake(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
        }

    request = {
        "model": "claude-haiku",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    }
    out = asyncio.run(
        proxy_anthropic_messages(
            request,
            dispatch_model="anthropic/claude-haiku-4-5",
            completion=fake,
        )
    )
    # dispatch_model overrides the inbound body's model.
    assert received["model"] == "anthropic/claude-haiku-4-5"
    assert received["max_tokens"] == 16
    assert out["type"] == "message"
    assert out["content"][0]["text"] == "hi"


@pytest.mark.unit
def test_proxy_anthropic_messages_coerces_pydantic_like_response() -> None:
    class _PydanticLike:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def model_dump(self) -> dict[str, Any]:
            return self._payload

    async def fake(**_: Any) -> _PydanticLike:
        return _PydanticLike({"type": "message", "role": "assistant", "content": []})

    out = asyncio.run(
        proxy_anthropic_messages(
            {
                "model": "x",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "x"}],
            },
            dispatch_model="anthropic/x",
            completion=fake,
        )
    )
    assert out["type"] == "message"


@pytest.mark.unit
def test_proxy_anthropic_messages_threads_api_key_and_headers() -> None:
    received: dict[str, Any] = {}

    async def fake(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return {"type": "message", "role": "assistant", "content": []}

    asyncio.run(
        proxy_anthropic_messages(
            {
                "model": "x",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "x"}],
            },
            dispatch_model="anthropic/x",
            completion=fake,
            forward_headers={
                "authorization": "Bearer xyz",
                "anthropic-beta": "feature-x",
                # Hop-by-hop / SDK-owned headers must be filtered.
                "content-type": "application/json",
                "content-length": "123",
            },
            api_key="explicit-key",
        )
    )
    assert received["api_key"] == "explicit-key"
    forwarded = received["extra_headers"]
    # Inbound ``authorization`` is stripped when api_key is explicit:
    # otherwise the openai-sdk lets extra_headers override the api_key
    # kwarg, leaking the inbound bearer to the upstream provider.
    assert "authorization" not in forwarded
    assert forwarded["anthropic-beta"] == "feature-x"
    assert "content-type" not in forwarded
    assert "content-length" not in forwarded
    # api_base omitted: kwargs must not carry the key, otherwise LiteLLM
    # would treat ``None`` as an explicit override of its provider default.
    assert "api_base" not in received


@pytest.mark.unit
def test_proxy_strips_inbound_x_api_key_when_api_key_explicit() -> None:
    """``x-api-key`` (Anthropic's inbound auth shape) also gets stripped.

    Symmetric to the ``authorization`` case: an inbound ``x-api-key`` from
    a claude-code-style client must not leak into ``extra_headers`` and
    overwrite the operator's chosen upstream key.
    """
    received: dict[str, Any] = {}

    async def fake(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return {"type": "message", "role": "assistant", "content": []}

    asyncio.run(
        proxy_anthropic_messages(
            {
                "model": "x",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "x"}],
            },
            dispatch_model="custom_openai/zai-org/GLM-5.1-FP8",
            completion=fake,
            forward_headers={
                "x-api-key": "sk-ant-from-claude-code",
                "anthropic-beta": "feature-x",
            },
            api_key="vk-vultr-from-env",
        )
    )
    assert received["api_key"] == "vk-vultr-from-env"
    forwarded = received["extra_headers"]
    assert "x-api-key" not in forwarded
    assert forwarded["anthropic-beta"] == "feature-x"


@pytest.mark.unit
def test_proxy_keeps_inbound_auth_when_api_key_unset() -> None:
    """When the rule has no ``api_key_env``, inbound auth is preserved.

    LiteLLM's per-provider env-var resolution (e.g. ``ANTHROPIC_API_KEY``)
    handles auth for the standard providers; in that mode the inbound
    bearer can legitimately survive into the upstream call so client-side
    OAuth flows (anthropic.com OAuth, etc.) still work.
    """
    received: dict[str, Any] = {}

    async def fake(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return {"type": "message", "role": "assistant", "content": []}

    asyncio.run(
        proxy_anthropic_messages(
            {
                "model": "x",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "x"}],
            },
            dispatch_model="anthropic/claude-x",
            completion=fake,
            forward_headers={"authorization": "Bearer client-oauth-token"},
            api_key=None,
        )
    )
    assert "api_key" not in received
    assert received["extra_headers"]["authorization"] == "Bearer client-oauth-token"


@pytest.mark.unit
def test_proxy_anthropic_messages_threads_api_base_to_litellm() -> None:
    """``api_base`` reaches the LiteLLM call so custom_openai etc. work.

    The auto-route path for openai-compatible third parties (Vultr, hosted
    vLLM) relies on this -- without ``api_base`` LiteLLM falls back to the
    provider-default URL (api.openai.com for ``custom_openai``).
    """
    received: dict[str, Any] = {}

    async def fake(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return {"type": "message", "role": "assistant", "content": []}

    asyncio.run(
        proxy_anthropic_messages(
            {
                "model": "x",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "x"}],
            },
            dispatch_model="custom_openai/zai-org/GLM-5.1-FP8",
            completion=fake,
            api_key="vk-test",
            api_base="https://api.vultrinference.com/v1",
        )
    )
    assert received["api_base"] == "https://api.vultrinference.com/v1"
    assert received["api_key"] == "vk-test"


@pytest.mark.unit
def test_stream_anthropic_messages_forwards_bytes_verbatim() -> None:
    chunks = [
        b'event: message_start\ndata: {"type": "message_start"}\n\n',
        b'event: content_block_delta\ndata: {"type": "content_block_delta"}\n\n',
        b'event: message_stop\ndata: {"type": "message_stop"}\n\n',
    ]

    async def fake_iter() -> Any:
        for chunk in chunks:
            yield chunk

    async def fake(**_: Any) -> Any:
        return fake_iter()

    request = {
        "model": "x",
        "max_tokens": 4,
        "messages": [{"role": "user", "content": "hi"}],
    }
    out = stream_anthropic_messages(request, dispatch_model="anthropic/x", completion=fake)

    async def collect() -> list[bytes]:
        return [chunk async for chunk in out]

    received = asyncio.run(collect())
    assert received == chunks


@pytest.mark.unit
def test_stream_anthropic_messages_forces_stream_true() -> None:
    received: dict[str, Any] = {}

    async def fake_iter() -> Any:
        for chunk in (b"event: x\ndata: {}\n\n",):
            yield chunk

    async def fake(**kwargs: Any) -> Any:
        received.update(kwargs)
        return fake_iter()

    request = {
        "model": "x",
        "max_tokens": 4,
        "messages": [{"role": "user", "content": "hi"}],
    }

    async def drain() -> None:
        async for _ in stream_anthropic_messages(
            request, dispatch_model="anthropic/x", completion=fake
        ):
            pass

    asyncio.run(drain())
    assert received["stream"] is True


@pytest.mark.unit
def test_stream_anthropic_messages_emits_error_event_on_dispatch_failure() -> None:
    async def boom(**_: Any) -> Any:
        raise RuntimeError("upstream exploded")

    request = {
        "model": "x",
        "max_tokens": 4,
        "messages": [{"role": "user", "content": "hi"}],
    }
    out = stream_anthropic_messages(request, dispatch_model="anthropic/x", completion=boom)

    async def collect() -> list[bytes]:
        return [chunk async for chunk in out]

    received = asyncio.run(collect())
    assert len(received) == 1
    text = received[0].decode()
    assert "event: error" in text
    assert "upstream exploded" in text
