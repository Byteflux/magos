"""``RouteDecision`` transport value. See ``docs/routing/pipeline.md``."""

from __future__ import annotations

from dataclasses import dataclass

from magos.registry.state import ModelEntry
from magos.routing.request import RoutedRequest
from magos.routing.schema import Action, Rule


def format_rule_label(rule: Rule, idx: int | None = None) -> str:
    """Stable identifier for logs: rule's ``name`` if set, else ``rule[idx]``.

    ``idx is None`` is rendered as ``rule[?]`` for diagnostics where the
    rule's position in the chain is unknown.
    """
    if rule.name is not None:
        return rule.name
    if idx is None:
        return "rule[?]"
    return f"rule[{idx}]"


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
        return format_rule_label(self.rule, idx)
