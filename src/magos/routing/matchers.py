"""Match-expression evaluator. See ``docs/routing/grammar.md``."""

from __future__ import annotations

import fnmatch
import functools
import re

from magos.registry.state import ModelEntry, RegistryState
from magos.routing.jq_compat import evaluate_predicate
from magos.routing.request import RoutedRequest
from magos.routing.schema import (
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
    ModelFieldAtom,
    ModelFieldExpr,
    Not,
    RegexMatcher,
)


def matches(  # noqa: PLR0911
    expr: MatchExpr,
    req: RoutedRequest,
    *,
    registry: RegistryState | None = None,
) -> bool:
    """True iff ``expr`` matches ``req``. ``model_field`` atoms need ``registry``; absent → False."""
    if isinstance(expr, AllOf):
        return all(matches(child, req, registry=registry) for child in expr.all_of)
    if isinstance(expr, AnyOf):
        return any(matches(child, req, registry=registry) for child in expr.any_of)
    if isinstance(expr, Not):
        return not matches(expr.not_, req, registry=registry)
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
    if isinstance(expr, ModelFieldAtom):
        return _model_field_matches(expr.model_field, req, registry)
    raise TypeError(f"unhandled MatchExpr variant: {type(expr).__name__}")


# Compile-cache: avoids per-request pattern compilation cost.
@functools.lru_cache(maxsize=512)
def _compile_regex(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


@functools.lru_cache(maxsize=512)
def _compile_glob(pattern: str) -> re.Pattern[str]:
    # fnmatch.translate produces a regex; compile once and reuse.
    return re.compile(fnmatch.translate(pattern))


def _matcher_matches(matcher: Matcher, value: str) -> bool:
    if isinstance(matcher, LiteralMatcher):
        return value == matcher.literal
    if isinstance(matcher, GlobMatcher):
        # Case-sensitive; opt into case-insensitive via regex (?i).
        return _compile_glob(matcher.glob).match(value) is not None
    if isinstance(matcher, RegexMatcher):
        # fullmatch so partial matches don't sneak through.
        return _compile_regex(matcher.regex).fullmatch(value) is not None
    raise TypeError(f"unhandled Matcher variant: {type(matcher).__name__}")


def _model_field_matches(
    expr: ModelFieldExpr,
    req: RoutedRequest,
    registry: RegistryState | None,
) -> bool:
    """Evaluate ``model_field`` against the resolved registry entry.

    Lookup: exact namespaced id then raw_id scan (single-match wins).
    """
    if registry is None:
        return False
    model = str(req.body.get("model", ""))
    if not model:
        return False
    entry = _resolve_entry(model, registry)
    if entry is None:
        return False
    return _apply_op(getattr(entry, expr.field), expr.op, expr.value)


def _resolve_entry(model: str, registry: RegistryState) -> ModelEntry | None:
    direct = registry.get(model)
    if direct is not None:
        return direct
    candidates = [e for e in registry.entries.values() if e.raw_id == model]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _apply_op(  # noqa: PLR0911
    field_value: object,
    op: str,
    value: int | float | str | list[int | float | str],
) -> bool:
    if field_value is None:
        return False
    if op == "contains":
        # Sequence-only; string fields use ``eq`` for membership.
        if not isinstance(field_value, (tuple, list)):
            return False
        return value in field_value
    if op == "in":
        if not isinstance(value, list):
            return False
        return field_value in value
    # Comparison ops require comparable scalars.
    if not isinstance(field_value, (int, float, str)) or not isinstance(value, (int, float, str)):
        return False
    if op == "eq":
        return field_value == value
    if op == "gt":
        return field_value > value  # type: ignore[operator]
    if op == "gte":
        return field_value >= value  # type: ignore[operator]
    if op == "lt":
        return field_value < value  # type: ignore[operator]
    if op == "lte":
        return field_value <= value  # type: ignore[operator]
    return False
