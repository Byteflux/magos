"""``Router``: ABC for the routing rule engine.

Implementations: :class:`RuleBasedRouter` (rule-based, the canonical engine),
:class:`AutoRouter` (registry-driven fallback), :class:`MeasuredRouter`
(decorator).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from magos.routing.decision import RouteDecision
from magos.routing.errors import RouteError
from magos.routing.request import RoutedRequest


class Router(ABC):
    """Decide what to do with a routed request."""

    @abstractmethod
    def route(self, req: RoutedRequest) -> RouteDecision | RouteError:
        """Resolve ``req`` into a decision or an error."""
