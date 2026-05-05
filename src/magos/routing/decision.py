"""``RouteDecision`` transport value. See ``docs/routing/pipeline.md``."""

from __future__ import annotations

from dataclasses import dataclass

from magos.registry.state import ModelEntry
from magos.routing.request import RoutedRequest
from magos.routing.schema import Action, Rule


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Successful route lookup. ``entry`` is set on auto-routed decisions only."""

    rule: Rule
    request: RoutedRequest
    dispatch_model: str
    entry: ModelEntry | None = None

    @property
    def action(self) -> Action:
        return self.rule.action

    @property
    def auto_routed(self) -> bool:
        return self.entry is not None

    def rule_label(self, idx: int | None = None) -> str:
        """Stable identifier for logs."""
        if self.rule.name is not None:
            return self.rule.name
        if idx is not None:
            return f"rule[{idx}]"
        return "rule[?]"
