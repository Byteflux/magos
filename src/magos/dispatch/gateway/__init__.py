"""`magos.dispatch.gateway`: `Gateway` ABC + canonical implementations.

Public surface:

- `Gateway` — ABC.
- `PassthroughGateway` — byte-exact httpx forward.
- `TranslateGateway` — LiteLLM SDK + CCR wrap.
- `CountTokensGateway` — count-tokens endpoint.
- `RoutedGateway` — composite selector wiring the three above.
- `MeasuredGateway` — decorator emitting OTel metrics per dispatch.
- `TracingGateway` — decorator opening an OTel span per dispatch.
"""

from __future__ import annotations

from magos.dispatch.gateway.base import Gateway
from magos.dispatch.gateway.count_tokens import CountTokensGateway
from magos.dispatch.gateway.measured import MeasuredGateway
from magos.dispatch.gateway.passthrough import PassthroughGateway
from magos.dispatch.gateway.routed import RoutedGateway
from magos.dispatch.gateway.tracing import TracingGateway
from magos.dispatch.gateway.translate import TranslateGateway

__all__ = [
    "CountTokensGateway",
    "Gateway",
    "MeasuredGateway",
    "PassthroughGateway",
    "RoutedGateway",
    "TracingGateway",
    "TranslateGateway",
]
