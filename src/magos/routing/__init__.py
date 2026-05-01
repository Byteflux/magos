"""Declarative rule-based routing.

Public surface: load a ``RoutingConfig`` from YAML and (in later slices) call
``route()`` on a ``RoutedRequest`` to obtain a ``RouteDecision`` or
``RouteError``. The matchers, rewrites, engine, and dispatcher are
implemented in subsequent slices.
"""

from __future__ import annotations

from magos.routing.loader import RoutingConfigError, load_config
from magos.routing.models import (
    Action,
    AddHeader,
    AllOf,
    AnyOf,
    EndpointAtom,
    GlobMatcher,
    HeaderAtom,
    HeaderPair,
    JqAtom,
    JqPatch,
    LiteralMatcher,
    Matcher,
    MatchExpr,
    ModelAtom,
    NamedValue,
    Not,
    RegexMatcher,
    RemoveHeader,
    Rewrite,
    RoutingConfig,
    Rule,
    SetHeader,
    SetModel,
)
from magos.routing.request import ENDPOINTS, Endpoint, RoutedRequest

__all__ = [
    "ENDPOINTS",
    "Action",
    "AddHeader",
    "AllOf",
    "AnyOf",
    "Endpoint",
    "EndpointAtom",
    "GlobMatcher",
    "HeaderAtom",
    "HeaderPair",
    "JqAtom",
    "JqPatch",
    "LiteralMatcher",
    "MatchExpr",
    "Matcher",
    "ModelAtom",
    "NamedValue",
    "Not",
    "RegexMatcher",
    "RemoveHeader",
    "Rewrite",
    "RoutedRequest",
    "RoutingConfig",
    "RoutingConfigError",
    "Rule",
    "SetHeader",
    "SetModel",
    "load_config",
]
