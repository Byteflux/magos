"""Tests for the shared ingress dispatch helper.

Two invariants:

1. `route()` runs off the asyncio event loop. Routing is sync by design
   and can do CPU-bound work or block on Headroom's Kompress thread-locked
   singleton during a cold-cache download. If the loop services that work
   directly, every in-flight request stalls until the blocking step
   completes, which is how the embedded mitm proxy disconnects clients on
   the first request after a cold start.
2. `request_id` is bound via `structlog.contextvars` at the API boundary
   so every per-request log line carries the same correlation id, including
   mid-stream events that fire after the handler returns.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import pytest
import structlog

from magos.api import build_api
from magos.routing import RouteError
from tests.api._helpers import translate_only_cfg


@pytest.mark.unit
def test_route_does_not_block_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two concurrent requests served by a sync-blocking `route()` must
    overlap in wall time, proving the dispatch helper offloads routing to
    a worker thread instead of running it on the event loop.
    """
    sleep_seconds = 0.4

    def slow_route(req: Any, *_args: Any, **_kwargs: Any) -> RouteError:
        time.sleep(sleep_seconds)
        return RouteError(
            status=404,
            code="unmatched",
            message="stub",
            model=str(req.body.get("model", "")),
            endpoint=req.endpoint,
        )

    monkeypatch.setattr(
        "magos.routing.engine.rule_based.RuleBasedRouter.route",
        lambda self, req: slow_route(req),
    )

    app = build_api(routing=translate_only_cfg())
    body = {"model": "x", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}

    async def fire_concurrent() -> float:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            started = time.perf_counter()
            await asyncio.gather(
                client.post("/v1/messages", json=body),
                client.post("/v1/messages", json=body),
            )
            return time.perf_counter() - started

    elapsed = asyncio.run(fire_concurrent())

    # If route() ran on the loop, the two requests would serialise to
    # ~2 * sleep_seconds. With offload, both block in worker threads in
    # parallel and the gather completes in ~sleep_seconds. Allow generous
    # headroom for scheduling jitter on slow CI.
    assert elapsed < sleep_seconds * 1.6, (
        f"requests serialised: {elapsed:.3f}s (expected < {sleep_seconds * 1.6:.3f}s); "
        "route() is blocking the asyncio event loop"
    )


@pytest.mark.unit
def test_request_id_bound_during_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """`X-Request-ID` from the inbound request is bound for every per-request log line."""
    seen: dict[str, Any] = {}

    def capture_route(_self: Any, req: Any) -> RouteError:
        seen["contextvars"] = dict(structlog.contextvars.get_contextvars())
        return RouteError(
            status=404,
            code="unmatched",
            message="stub",
            model=str(req.body.get("model", "")),
            endpoint=req.endpoint,
        )

    monkeypatch.setattr(
        "magos.routing.engine.rule_based.RuleBasedRouter.route",
        capture_route,
    )

    app = build_api(routing=translate_only_cfg())
    body = {"model": "x", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}

    async def go() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post("/v1/messages", json=body, headers={"X-Request-ID": "abc-123"})

    asyncio.run(go())

    assert seen["contextvars"].get("request_id") == "abc-123"


@pytest.mark.unit
def test_request_id_generated_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without `X-Request-ID`, a 12-char hex token is generated."""
    seen: dict[str, Any] = {}

    def capture_route(_self: Any, req: Any) -> RouteError:
        seen["contextvars"] = dict(structlog.contextvars.get_contextvars())
        return RouteError(
            status=404,
            code="unmatched",
            message="stub",
            model=str(req.body.get("model", "")),
            endpoint=req.endpoint,
        )

    monkeypatch.setattr(
        "magos.routing.engine.rule_based.RuleBasedRouter.route",
        capture_route,
    )

    app = build_api(routing=translate_only_cfg())
    body = {"model": "x", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}

    async def go() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post("/v1/messages", json=body)

    asyncio.run(go())

    request_id = seen["contextvars"].get("request_id")
    assert isinstance(request_id, str)
    assert len(request_id) == 12
    assert all(c in "0123456789abcdef" for c in request_id)
