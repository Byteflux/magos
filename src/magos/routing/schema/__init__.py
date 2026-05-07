"""Pydantic schemas for declarative routing config. See ``docs/routing/grammar.md``.

Three concerns split across siblings:

- :mod:`grammar` — match grammar: matchers, atoms, combinators, ``MatchExpr``.
- :mod:`rewrites` — rewrite primitives, ``CompressOptions``, ``Rewrite`` union.
- :mod:`structure` — top-level: ``Target``, ``Rule``, ``GuardedRewrites``,
  ``RoutingConfig``.

``config_uses_compress`` lives here since it walks the assembled config.
"""

from __future__ import annotations

from .grammar import (
    AllOf,
    AnyOf,
    EndpointAtom,
    GlobMatcher,
    HeaderAtom,
    HeaderPair,
    JqAtom,
    LiteralMatcher,
    Matcher,
    MatchExpr,
    ModelAtom,
    ModelFieldAtom,
    ModelFieldExpr,
    ModelFieldOp,
    Not,
    RegexMatcher,
)
from .rewrites import (
    AddHeader,
    Compress,
    CompressMode,
    CompressOptions,
    JqPatch,
    NamedValue,
    RemoveHeader,
    Rewrite,
    SetHeader,
    SetModel,
)
from .structure import (
    AuthHeaderShape,
    GatewayMode,
    GuardedRewrites,
    PreRewrite,
    RoutingConfig,
    Rule,
    Target,
)


def config_uses_compress(cfg: RoutingConfig) -> bool:
    """True iff any pre-rewrite or rule rewrite is a ``compress`` op.

    Walks into ``GuardedRewrites`` so a guarded pre-rewrite still
    counts; the engine evaluates them at request time.
    """
    for entry in cfg.pre_rewrites:
        if isinstance(entry, Compress):
            return True
        if isinstance(entry, GuardedRewrites) and any(
            isinstance(rw, Compress) for rw in entry.rewrites
        ):
            return True
    return any(isinstance(rw, Compress) for rule in cfg.rules for rw in rule.rewrites)


__all__ = [
    "AddHeader",
    "AllOf",
    "AnyOf",
    "AuthHeaderShape",
    "Compress",
    "CompressMode",
    "CompressOptions",
    "EndpointAtom",
    "GatewayMode",
    "GlobMatcher",
    "GuardedRewrites",
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
    "PreRewrite",
    "RegexMatcher",
    "RemoveHeader",
    "Rewrite",
    "RoutingConfig",
    "Rule",
    "SetHeader",
    "SetModel",
    "Target",
    "config_uses_compress",
]
