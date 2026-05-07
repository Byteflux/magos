"""`TracingGateway`: span opens with expected attributes per dispatch."""

from __future__ import annotations

import asyncio
from typing import Any

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import magos.dispatch.gateway.tracing as tracing_mod
from magos.dispatch.gateway import (
    CountTokensGateway,
    PassthroughGateway,
    RoutedGateway,
    TracingGateway,
    TranslateGateway,
)
from magos.routing import RoutingConfig
from magos.routing.decision import RouteDecision
from magos.routing.engine import route
from magos.routing.request import RoutedRequest


def _decision() -> RouteDecision:
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
    return decision


def _inner_gateway() -> RoutedGateway:
    return RoutedGateway(
        passthrough=PassthroughGateway(),
        translate=TranslateGateway(),
        count_tokens=CountTokensGateway(),
    )


def test_tracing_gateway_opens_span_with_expected_attributes() -> None:
    """TracingGateway opens a `gateway.dispatch` span with target attributes."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Patch the module-level tracer so TracingGateway uses our test provider.
    original_tracer = tracing_mod._tracer
    tracing_mod._tracer = provider.get_tracer("magos.gateway.test")
    try:
        gateway = TracingGateway(_inner_gateway())
        decision = _decision()

        async def fake_completion(**_: Any) -> Any:
            return {
                "model": "gpt-4o",
                "choices": [{"message": {"role": "assistant", "content": "hi"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

        asyncio.run(gateway.dispatch(decision, completion=fake_completion))
    finally:
        tracing_mod._tracer = original_tracer

    spans = exporter.get_finished_spans()
    assert spans, "Expected at least one span"
    span = spans[0]
    assert span.name == "gateway.dispatch"
    attrs = dict(span.attributes or {})
    assert attrs.get("magos.gateway") == "translate"
    assert attrs.get("magos.provider") == "openai"
    assert attrs.get("magos.endpoint") == "/v1/chat/completions"
    assert "magos.dispatch_model" in attrs


def test_tracing_gateway_dispatch_succeeds_without_configured_tracing() -> None:
    """TracingGateway is a no-op when no real tracer is configured."""
    gateway = TracingGateway(_inner_gateway())
    decision = _decision()

    async def fake_completion(**_: Any) -> Any:
        return {
            "model": "gpt-4o",
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    # Should not raise; the no-op tracer accepts the span silently.
    result = asyncio.run(gateway.dispatch(decision, completion=fake_completion))
    assert result is not None
