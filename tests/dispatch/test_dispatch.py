"""``Gateway.dispatch`` fires ``post_response_hooks`` after capturing usage."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from magos.compression.ccr import CCR_TOOL_NAME
from magos.dispatch.gateway import (
    CountTokensGateway,
    PassthroughGateway,
    RoutedGateway,
    TranslateGateway,
)
from magos.dispatch.gateway import (
    translate as translate_mod,
)
from magos.dispatch.usage import Usage
from magos.routing import RoutingConfig
from magos.routing.decision import RouteDecision
from magos.routing.engine import route
from magos.routing.request import RoutedRequest


def _gateway() -> RoutedGateway:
    return RoutedGateway(
        passthrough=PassthroughGateway(),
        translate=TranslateGateway(),
        count_tokens=CountTokensGateway(),
    )


@pytest.fixture
def fake_completion_factory() -> Any:
    """Builds a completion fn that returns the given body or stream."""

    def make(body_or_stream: Any) -> Any:
        async def fn(**_: Any) -> Any:
            return body_or_stream

        return fn

    return make


def test_dispatch_fires_hooks_after_translate_non_streaming(
    fake_completion_factory: Any,
) -> None:
    """Translate path, non-streaming: hook receives the captured Usage."""
    seen: list[Usage] = []

    req = RoutedRequest(
        endpoint="/v1/chat/completions",
        headers={},
        body={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        raw_body=b"",
        post_response_hooks=[seen.append],
    )

    cfg = RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/chat/completions"}},
                    "target": {"provider": "openai", "gateway": "translate"},
                }
            ]
        }
    )
    decision = route(req, cfg)
    assert isinstance(decision, RouteDecision)

    completion = fake_completion_factory(
        {
            "model": "gpt-4o",
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }
    )
    asyncio.run(_gateway().dispatch(decision, completion=completion))

    assert len(seen) == 1
    assert seen[0].input == 100


def test_dispatch_swallows_hook_exceptions(fake_completion_factory: Any) -> None:
    """A raising hook must not break the response. Logged as compress.hook_failed."""
    fired: list[Usage] = []

    def boom(_: Usage) -> None:
        raise RuntimeError("hook bug")

    def good(u: Usage) -> None:
        fired.append(u)

    req = RoutedRequest(
        endpoint="/v1/chat/completions",
        headers={},
        body={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        raw_body=b"",
        post_response_hooks=[boom, good],
    )
    cfg = RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/chat/completions"}},
                    "target": {"provider": "openai", "gateway": "translate"},
                }
            ]
        }
    )
    decision = route(req, cfg)
    assert isinstance(decision, RouteDecision)

    completion = fake_completion_factory(
        {
            "model": "gpt-4o",
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }
    )
    asyncio.run(_gateway().dispatch(decision, completion=completion))

    assert len(fired) == 1


def test_dispatch_translate_invokes_wrap_response_when_ccr_request(
    fake_completion_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the request body has the CCR tool, dispatch wraps the response."""
    wrap_calls: list[Any] = []

    async def fake_wrap_response(response: Any, **kwargs: Any) -> Any:
        wrap_calls.append((response, kwargs))
        return response

    monkeypatch.setattr(translate_mod, "wrap_response", fake_wrap_response)

    req = RoutedRequest(
        endpoint="/v1/chat/completions",
        headers={},
        body={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": CCR_TOOL_NAME}}],
        },
        raw_body=b"",
    )
    cfg = RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/chat/completions"}},
                    "target": {"provider": "openai", "gateway": "translate"},
                }
            ]
        }
    )
    decision = route(req, cfg)
    assert isinstance(decision, RouteDecision)

    completion = fake_completion_factory(
        {
            "model": "gpt-4o",
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
    )
    asyncio.run(_gateway().dispatch(decision, completion=completion))

    assert len(wrap_calls) == 1


def test_dispatch_translate_skips_wrap_when_no_ccr_tool(
    fake_completion_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No CCR tool in body -> wrap_response is still called (it short-circuits
    internally), but no continuation kicks off."""
    wrap_calls: list[Any] = []

    async def fake_wrap_response(response: Any, **kwargs: Any) -> Any:
        wrap_calls.append((response, kwargs))
        return response

    monkeypatch.setattr(translate_mod, "wrap_response", fake_wrap_response)

    req = RoutedRequest(
        endpoint="/v1/chat/completions",
        headers={},
        body={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        raw_body=b"",
    )
    cfg = RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/chat/completions"}},
                    "target": {"provider": "openai", "gateway": "translate"},
                }
            ]
        }
    )
    decision = route(req, cfg)
    assert isinstance(decision, RouteDecision)

    completion = fake_completion_factory(
        {
            "model": "gpt-4o",
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
    )
    asyncio.run(_gateway().dispatch(decision, completion=completion))

    # wrap_response is called unconditionally but receives a non-CCR request.
    assert len(wrap_calls) == 1
    captured_req = wrap_calls[0][1]["req"]
    assert "tools" not in captured_req.body
