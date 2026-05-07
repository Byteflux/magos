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

from .base import Gateway
from .count_tokens import CountTokensGateway
from .measured import MeasuredGateway
from .passthrough import PassthroughGateway
from .routed import RoutedGateway
from .tracing import TracingGateway
from .translate import TranslateGateway

__all__ = [
    "CountTokensGateway",
    "Gateway",
    "MeasuredGateway",
    "PassthroughGateway",
    "RoutedGateway",
    "TracingGateway",
    "TranslateGateway",
]
