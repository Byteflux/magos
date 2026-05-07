"""Pipes-and-Filters step abstraction for the routing pipeline.

``Transform`` is the single sync ABC that all rewrite and compression steps
implement. Phase C3b will flip ``apply`` to async; until then callers use the
sync form and the router applies transforms at decision time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from magos.registry.state import RegistryState
from magos.routing.request import RoutedRequest


class Transform(ABC):
    """Pure transform on a ``RoutedRequest``. Pipes-and-Filters step.

    Sync today; Phase C3b will flip to async.
    """

    @abstractmethod
    def apply(
        self,
        req: RoutedRequest,
        *,
        registry: RegistryState | None = None,
    ) -> RoutedRequest: ...
