"""Declarative rule-based routing.

Public surface: load a ``RoutingConfig`` from YAML, then call ``route()`` on
a ``RoutedRequest`` to obtain a ``RouteDecision`` or ``RouteError``.
``magos.egress.dispatch`` consumes the decision and bridges to the
``magos.egress.translate`` and ``magos.egress.passthrough`` modules.
"""

from __future__ import annotations

from magos.routing.engine import (
    RouteDecision,
    apply_post_rewrites,
    apply_pre_rewrites,
    route,
)
from magos.routing.errors import (
    RouteError,
    error_envelope,
    format_dispatch_error_message,
    format_unmatched_message,
)
from magos.routing.loader import RoutingConfigError, load_config
from magos.routing.matchers import matches
from magos.routing.request import ENDPOINTS, Endpoint, RoutedRequest
from magos.routing.rewrites import RewriteError, apply_rewrites
from magos.routing.schema import (
    Action,
    AddHeader,
    AllOf,
    AnyOf,
    Compress,
    CompressMode,
    CompressOptions,
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
    ModelFieldAtom,
    ModelFieldExpr,
    ModelFieldOp,
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

__all__ = [
    "ENDPOINTS",
    "Action",
    "AddHeader",
    "AllOf",
    "AnyOf",
    "Compress",
    "CompressMode",
    "CompressOptions",
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
    "ModelFieldAtom",
    "ModelFieldExpr",
    "ModelFieldOp",
    "NamedValue",
    "Not",
    "RegexMatcher",
    "RemoveHeader",
    "Rewrite",
    "RewriteError",
    "RouteDecision",
    "RouteError",
    "RoutedRequest",
    "RoutingConfig",
    "RoutingConfigError",
    "Rule",
    "SetHeader",
    "SetModel",
    "apply_post_rewrites",
    "apply_pre_rewrites",
    "apply_rewrites",
    "error_envelope",
    "format_dispatch_error_message",
    "format_unmatched_message",
    "load_config",
    "matches",
    "route",
]
