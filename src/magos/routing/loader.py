"""Load and validate ``magos.yaml`` routing config.

Validation splits across pydantic (structural — types, single-key unions,
required fields) and post-load checks here:

- regex patterns compile via ``re.compile``
- glob patterns translate via ``fnmatch.translate``
- jq programs compile via ``jq.compile``
- ``count_tokens_mode: passthrough`` requires the action's provider to be
  registered in ``magos.tokens.PASSTHROUGH_DISPATCH``
- a structlog warning fires per rule whose ``mode`` is ``passthrough`` and
  whose body-touching rewrites (post or pre) would force re-serialisation,
  breaking prompt-cache byte-exactness.
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

import yaml
from pydantic import ValidationError

from magos.obs import get_logger
from magos.routing.jq_compat import JqCompileError, check_program
from magos.routing.models import (
    AllOf,
    AnyOf,
    EndpointAtom,
    GlobMatcher,
    HeaderAtom,
    JqAtom,
    JqPatch,
    Matcher,
    MatchExpr,
    ModelAtom,
    Not,
    RegexMatcher,
    Rewrite,
    RoutingConfig,
    Rule,
    SetModel,
)
from magos.tokens import PASSTHROUGH_DISPATCH

log = get_logger("magos.routing.loader")


class RoutingConfigError(ValueError):
    """Raised on post-load validation failures (semantic, not structural)."""


def load_config(path: str | Path) -> RoutingConfig:
    """Read ``path``, parse YAML, validate, and return ``RoutingConfig``."""
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    if not isinstance(data, dict):
        raise RoutingConfigError(
            f"{p}: top-level YAML must be a mapping, got {type(data).__name__}"
        )
    try:
        cfg = RoutingConfig.model_validate(data)
    except ValidationError as exc:
        raise RoutingConfigError(f"{p}: invalid routing config: {exc}") from exc
    _validate_compiled(cfg, source=str(p))
    _validate_count_tokens(cfg, source=str(p))
    _warn_passthrough_body_touch(cfg)
    return cfg


def _rule_label(rule: Rule, idx: int) -> str:
    return rule.name or f"rule[{idx}]"


def _iter_match_atoms(expr: MatchExpr) -> Iterator[MatchExpr]:
    """Yield every leaf atom under ``expr`` (combinators recurse)."""
    if isinstance(expr, AllOf):
        for child in expr.all_of:
            yield from _iter_match_atoms(child)
    elif isinstance(expr, AnyOf):
        for child in expr.any_of:
            yield from _iter_match_atoms(child)
    elif isinstance(expr, Not):
        yield from _iter_match_atoms(expr.not_)
    else:
        yield expr


def _iter_matchers(expr: MatchExpr) -> Iterator[Matcher]:
    """Yield every non-jq matcher value (regex/glob/literal) under ``expr``."""
    for atom in _iter_match_atoms(expr):
        if isinstance(atom, ModelAtom):
            yield atom.model
        elif isinstance(atom, EndpointAtom):
            yield atom.endpoint
        elif isinstance(atom, HeaderAtom):
            yield atom.header.name
            yield atom.header.value


def _iter_jq_atoms(expr: MatchExpr) -> Iterator[JqAtom]:
    for atom in _iter_match_atoms(expr):
        if isinstance(atom, JqAtom):
            yield atom


def _validate_compiled(cfg: RoutingConfig, *, source: str) -> None:
    """Compile every regex, glob, and jq program; raise on first failure.

    Errors include the rule label and the offending pattern so the operator
    can fix the YAML without grepping the stack trace.
    """
    for idx, rule in enumerate(cfg.rules):
        label = _rule_label(rule, idx)
        for matcher in _iter_matchers(rule.match):
            _check_matcher(matcher, where=f"{source}: {label} match")
        for atom in _iter_jq_atoms(rule.match):
            try:
                check_program(atom.jq)
            except JqCompileError as exc:
                raise RoutingConfigError(f"{source}: {label} match: {exc}") from exc
        for r_idx, rw in enumerate(rule.rewrites):
            _check_rewrite(rw, where=f"{source}: {label} rewrites[{r_idx}]")
    for r_idx, rw in enumerate(cfg.pre_rewrites):
        _check_rewrite(rw, where=f"{source}: pre_rewrites[{r_idx}]")


def _check_matcher(matcher: Matcher, *, where: str) -> None:
    if isinstance(matcher, RegexMatcher):
        try:
            re.compile(matcher.regex)
        except re.error as exc:
            raise RoutingConfigError(f"{where}: invalid regex {matcher.regex!r}: {exc}") from exc
    elif isinstance(matcher, GlobMatcher):
        # ``fnmatch.translate`` returns a regex string; compile it to surface
        # any glob-syntax error as a real exception. Stable across CPython.
        try:
            re.compile(fnmatch.translate(matcher.glob))
        except re.error as exc:
            raise RoutingConfigError(f"{where}: invalid glob {matcher.glob!r}: {exc}") from exc


def _check_rewrite(rw: Rewrite, *, where: str) -> None:
    if isinstance(rw, JqPatch):
        try:
            check_program(rw.jq_patch)
        except JqCompileError as exc:
            raise RoutingConfigError(f"{where}: {exc}") from exc


def _validate_count_tokens(cfg: RoutingConfig, *, source: str) -> None:
    """Reject ``count_tokens_mode: passthrough`` for unsupported providers."""
    supported = frozenset(PASSTHROUGH_DISPATCH.keys())
    for idx, rule in enumerate(cfg.rules):
        if rule.action.count_tokens_mode != "passthrough":
            continue
        if rule.action.provider not in supported:
            label = _rule_label(rule, idx)
            raise RoutingConfigError(
                f"{source}: {label}: count_tokens_mode='passthrough' is not "
                f"implemented for provider={rule.action.provider!r} "
                f"(supported: {sorted(supported)})"
            )


def _rewrites_touch_body(rewrites: Iterable[Rewrite]) -> bool:
    return any(isinstance(rw, (SetModel, JqPatch)) for rw in rewrites)


def _warn_passthrough_body_touch(cfg: RoutingConfig) -> None:
    """Warn per rule that combines body-touching rewrites with passthrough.

    Header-only rewrites (``set_header``/``add_header``/``remove_header``)
    are silent; only ``set_model`` and ``jq_patch`` force re-serialisation
    of the body and break prompt-cache byte-exactness.
    """
    pre_touches = _rewrites_touch_body(cfg.pre_rewrites)
    for idx, rule in enumerate(cfg.rules):
        if rule.action.mode != "passthrough":
            continue
        post_touches = _rewrites_touch_body(rule.rewrites)
        if pre_touches or post_touches:
            log.warning(
                "routing.passthrough_body_touch",
                rule=_rule_label(rule, idx),
                pre_rewrites_touch=pre_touches,
                post_rewrites_touch=post_touches,
                hint="prompt cache byte-exactness will be broken for this rule",
            )
