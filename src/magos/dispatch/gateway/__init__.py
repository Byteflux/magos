"""``magos.dispatch.gateway``: ``Gateway`` ABC + canonical implementations.

Public surface:

- :class:`Gateway` — ABC.
- :class:`PassthroughGateway` — byte-exact httpx forward.
- :class:`TranslateGateway` — LiteLLM SDK + CCR wrap.
- :class:`CountTokensGateway` — count-tokens endpoint.
- :class:`RoutedGateway` — composite selector wiring the three above.
- :class:`MeasuredGateway` — decorator emitting OTel metrics per dispatch.
- :class:`TracingGateway` — decorator opening an OTel span per dispatch.
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
