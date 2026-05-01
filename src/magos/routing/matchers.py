"""Pure evaluator for routing match expressions.

Stateless: every call recompiles regex/jq programs. Slice 4 will introduce
a per-rule compiled-artifact cache in the engine; for now ``re.compile`` is
itself cached by the stdlib and ``jq.compile`` cost is acceptable for the
test surface. The matcher module exposes only ``matches``; everything else
is private dispatch helpers.
"""

from __future__ import annotations

import fnmatch
import re

from magos.routing.jq_compat import evaluate_predicate
from magos.routing.models import (
    AllOf,
    AnyOf,
    EndpointAtom,
    GlobMatcher,
    HeaderAtom,
    JqAtom,
    LiteralMatcher,
    Matcher,
    MatchExpr,
    ModelAtom,
    Not,
    RegexMatcher,
)
from magos.routing.request import RoutedRequest


def matches(expr: MatchExpr, req: RoutedRequest) -> bool:  # noqa: PLR0911
    """True iff ``expr`` matches ``req``.

    Exhaustive isinstance dispatch over the closed ``MatchExpr`` union; one
    branch per variant is the readable shape, so the per-function return cap
    is suppressed.
    """
    if isinstance(expr, AllOf):
        return all(matches(child, req) for child in expr.all_of)
    if isinstance(expr, AnyOf):
        return any(matches(child, req) for child in expr.any_of)
    if isinstance(expr, Not):
        return not matches(expr.not_, req)
    if isinstance(expr, ModelAtom):
        model = str(req.body.get("model", ""))
        return _matcher_matches(expr.model, model)
    if isinstance(expr, EndpointAtom):
        return _matcher_matches(expr.endpoint, req.endpoint)
    if isinstance(expr, HeaderAtom):
        return any(
            _matcher_matches(expr.header.name, name) and _matcher_matches(expr.header.value, value)
            for name, value in req.headers.items()
        )
    if isinstance(expr, JqAtom):
        return evaluate_predicate(expr.jq, dict(req.body))
    raise TypeError(f"unhandled MatchExpr variant: {type(expr).__name__}")


def _matcher_matches(matcher: Matcher, value: str) -> bool:
    if isinstance(matcher, LiteralMatcher):
        return value == matcher.literal
    if isinstance(matcher, GlobMatcher):
        # Case-sensitive glob; users opt into case-insensitive via regex (?i).
        return fnmatch.fnmatchcase(value, matcher.glob)
    if isinstance(matcher, RegexMatcher):
        # fullmatch (not search) so partial matches don't sneak through; users
        # who want substring matching write ``.*foo.*`` explicitly.
        return re.fullmatch(matcher.regex, value) is not None
    raise TypeError(f"unhandled Matcher variant: {type(matcher).__name__}")
