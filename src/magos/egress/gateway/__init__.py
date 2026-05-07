"""``magos.egress.gateway``: ``Gateway`` ABC + canonical implementations.

Public surface:

- :class:`Gateway` — ABC.
- :class:`PassthroughGateway` — byte-exact httpx forward.
- :class:`TranslateGateway` — LiteLLM SDK + CCR wrap.
- :class:`CountTokensGateway` — count-tokens endpoint.
- :class:`RoutedGateway` — composite selector wiring the three above.
"""

from __future__ import annotations

from .base import Gateway
from .count_tokens import CountTokensGateway
from .passthrough import PassthroughGateway
from .routed import RoutedGateway
from .translate import TranslateGateway

__all__ = [
    "CountTokensGateway",
    "Gateway",
    "PassthroughGateway",
    "RoutedGateway",
    "TranslateGateway",
]
