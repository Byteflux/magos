"""Tests for the surviving token-counting strategies in ``magos.tokens``.

The orchestrator and provider-resolution table moved to the routing layer;
the strategies (``count_locally``, ``_anthropic_passthrough``) and the
registry (``PASSTHROUGH_DISPATCH``) stay here and are unit-tested below.
"""

from __future__ import annotations

import asyncio
from typing import Any

import anthropic
import pytest

from magos import tokens

SIMPLE_REQUEST: dict[str, Any] = {
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 16,
    "messages": [{"role": "user", "content": "Hello, world."}],
}


@pytest.mark.unit
def test_count_locally_returns_positive_int() -> None:
    n = tokens.count_locally(SIMPLE_REQUEST)
    assert isinstance(n, int)
    assert n > 0


@pytest.mark.unit
def test_count_locally_works_without_max_tokens() -> None:
    """count_tokens body has no ``max_tokens`` field per Anthropic's spec."""
    body = {k: v for k, v in SIMPLE_REQUEST.items() if k != "max_tokens"}
    assert "max_tokens" not in body
    n = tokens.count_locally(body)
    assert n == tokens.count_locally(SIMPLE_REQUEST)


@pytest.mark.unit
def test_passthrough_dispatch_registers_anthropic() -> None:
    assert "anthropic" in tokens.PASSTHROUGH_DISPATCH
    assert callable(tokens.PASSTHROUGH_DISPATCH["anthropic"])


@pytest.mark.unit
def test_anthropic_passthrough_invokes_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_anthropic_passthrough`` calls ``messages.count_tokens`` on the SDK."""
    captured: dict[str, Any] = {}

    class _FakeResult:
        input_tokens = 42

    class _FakeMessages:
        async def count_tokens(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return _FakeResult()

    class _FakeAnthropic:
        def __init__(self) -> None:
            self.messages = _FakeMessages()

        async def __aenter__(self) -> _FakeAnthropic:
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

    monkeypatch.setattr(anthropic, "AsyncAnthropic", _FakeAnthropic)
    n = asyncio.run(
        tokens._anthropic_passthrough(
            SIMPLE_REQUEST,
            forward_headers={
                "authorization": "Bearer x",
                "x-api-key": "should-be-filtered",
                "content-type": "application/json",
                "anthropic-beta": "feature-x",
                "anthropic-version": "2023-06-01",
            },
        )
    )
    assert n == 42
    assert captured["model"] == "claude-3-5-sonnet-20241022"
    # Only ``anthropic-*`` knobs survive: auth duplicates the SDK's own
    # header, and transport headers (content-type/accept/etc) interfere
    # with the SDK's HTTP machinery.
    assert captured["extra_headers"] == {
        "anthropic-beta": "feature-x",
        "anthropic-version": "2023-06-01",
    }
