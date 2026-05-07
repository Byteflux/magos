"""Pydantic schemas for declarative routing config. See ``docs/routing/grammar.md``.

Three concerns split across siblings:

- :mod:`grammar` — match grammar: matchers, atoms, combinators, ``MatchExpr``.
- :mod:`rewrites` — transform primitives (``SetModel``, ``SetHeader``, etc.), ``CompressOptions``.
- :mod:`structure` — top-level: ``Target``, ``Rule``, ``GuardedTransforms``,
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
    SetHeader,
    SetModel,
)
from .structure import (
    AuthHeaderShape,
    GatewayMode,
    GuardedTransforms,
    PreTransform,
    RoutingConfig,
    Rule,
    Target,
)


def config_uses_compress(cfg: RoutingConfig) -> bool:
    """True iff any pre-transform or rule transform is a ``compress`` op.

    Walks into ``GuardedTransforms`` so a guarded pre-transform still
    counts; the engine evaluates them at request time.
    """
    for entry in cfg.pre_transforms:
        if isinstance(entry, Compress):
            return True
        if isinstance(entry, GuardedTransforms) and any(
            isinstance(rw, Compress) for rw in entry.transforms
        ):
            return True
    return any(isinstance(rw, Compress) for rule in cfg.rules for rw in rule.transforms)


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
    "GuardedTransforms",
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
    "PreTransform",
    "RegexMatcher",
    "RemoveHeader",
    "RoutingConfig",
    "Rule",
    "SetHeader",
    "SetModel",
    "Target",
    "config_uses_compress",
]
