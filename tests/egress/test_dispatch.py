"""``dispatch_decision`` fires ``post_response_hooks`` after capturing usage."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from magos.egress.usage import Usage


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
    from magos.egress.dispatch import dispatch_decision  # noqa: PLC0415
    from magos.routing import RoutingConfig  # noqa: PLC0415
    from magos.routing.decision import RouteDecision  # noqa: PLC0415
    from magos.routing.engine import route  # noqa: PLC0415
    from magos.routing.request import RoutedRequest  # noqa: PLC0415

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
                    "action": {"provider": "openai", "mode": "translate"},
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
    asyncio.run(dispatch_decision(decision, completion=completion))

    assert len(seen) == 1
    assert seen[0].input == 100


def test_dispatch_swallows_hook_exceptions(fake_completion_factory: Any) -> None:
    """A raising hook must not break the response. Logged as compress.hook_failed."""
    from magos.egress.dispatch import dispatch_decision  # noqa: PLC0415
    from magos.routing import RoutingConfig  # noqa: PLC0415
    from magos.routing.decision import RouteDecision  # noqa: PLC0415
    from magos.routing.engine import route  # noqa: PLC0415
    from magos.routing.request import RoutedRequest  # noqa: PLC0415

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
                    "action": {"provider": "openai", "mode": "translate"},
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
    asyncio.run(dispatch_decision(decision, completion=completion))

    assert len(fired) == 1
