"""Load and validate `magos.yaml` routing config. See `docs/routing/errors.md`."""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

import yaml
from pydantic import ValidationError

from magos.routing.decision import format_rule_label
from magos.routing.jq_compat import JqCompileError, check_program
from magos.routing.schema import (
    AllOf,
    AnyOf,
    Compress,
    EndpointAtom,
    GlobMatcher,
    GuardedTransforms,
    HeaderAtom,
    JqAtom,
    JqPatch,
    Matcher,
    MatchExpr,
    ModelAtom,
    Not,
    PreTransform,
    RegexMatcher,
    RoutingConfig,
    SetModel,
)
from magos.telemetry import get_logger

log = get_logger("magos.routing.loader")


class RoutingConfigError(ValueError):
    """Raised on post-load validation failures (semantic, not structural)."""


# Top-level extras tolerated so the same YAML can carry registry blocks
# (`providers:` etc.); `RoutingConfig` itself stays `extra="forbid"`.
_ROUTING_KEYS: frozenset[str] = frozenset({"pre_transforms", "rules"})


def load_config(path: str | Path) -> RoutingConfig:
    """Read `path`, parse YAML, validate, and return `RoutingConfig`."""
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    if not isinstance(data, dict):
        raise RoutingConfigError(
            f"{p}: top-level YAML must be a mapping, got {type(data).__name__}"
        )
    return load_from_data(data, source=str(p))


def load_from_data(data: dict[str, object], *, source: str) -> RoutingConfig:
    """Validate a pre-parsed YAML mapping into `RoutingConfig`.

    Used by `magos.config.loader.load_full_config` so the source
    file is parsed once and shared across the routing / registry /
    ingress parsers.
    """
    routing_subset = {k: v for k, v in data.items() if k in _ROUTING_KEYS}
    try:
        cfg = RoutingConfig.model_validate(routing_subset)
    except ValidationError as exc:
        raise RoutingConfigError(f"{source}: invalid routing config: {exc}") from exc
    _validate_compiled(cfg, source=source)
    _validate_passthrough_base_url(cfg, source=source)
    _warn_passthrough_body_touch(cfg)
    return cfg


def _iter_match_atoms(expr: MatchExpr) -> Iterator[MatchExpr]:
    """Yield every leaf atom under `expr` (combinators recurse)."""
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
    """Yield every non-jq matcher value (regex/glob/literal) under `expr`."""
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
    """Compile every regex, glob, and jq program; raise on first failure with rule label."""
    for idx, rule in enumerate(cfg.rules):
        label = format_rule_label(rule, idx)
        for matcher in _iter_matchers(rule.match):
            _check_matcher(matcher, where=f"{source}: {label} match")
        for atom in _iter_jq_atoms(rule.match):
            try:
                check_program(atom.jq)
            except JqCompileError as exc:
                raise RoutingConfigError(f"{source}: {label} match: {exc}") from exc
        for r_idx, rw in enumerate(rule.transforms):
            _check_rewrite(rw, where=f"{source}: {label} transforms[{r_idx}]")
    for r_idx, entry in enumerate(cfg.pre_transforms):
        where = f"{source}: pre_transforms[{r_idx}]"
        if isinstance(entry, GuardedTransforms):
            for matcher in _iter_matchers(entry.match):
                _check_matcher(matcher, where=f"{where} match")
            for atom in _iter_jq_atoms(entry.match):
                try:
                    check_program(atom.jq)
                except JqCompileError as exc:
                    raise RoutingConfigError(f"{where} match: {exc}") from exc
            for inner_idx, rw in enumerate(entry.transforms):
                _check_rewrite(rw, where=f"{where} transforms[{inner_idx}]")
        else:
            _check_rewrite(entry, where=where)


def _check_matcher(matcher: Matcher, *, where: str) -> None:
    if isinstance(matcher, RegexMatcher):
        try:
            re.compile(matcher.regex)
        except re.error as exc:
            raise RoutingConfigError(f"{where}: invalid regex {matcher.regex!r}: {exc}") from exc
    elif isinstance(matcher, GlobMatcher):
        # Translate to regex and compile to surface glob-syntax errors.
        try:
            re.compile(fnmatch.translate(matcher.glob))
        except re.error as exc:
            raise RoutingConfigError(f"{where}: invalid glob {matcher.glob!r}: {exc}") from exc


def _check_rewrite(rw: object, *, where: str) -> None:
    if isinstance(rw, JqPatch):
        try:
            check_program(rw.jq_patch)
        except JqCompileError as exc:
            raise RoutingConfigError(f"{where}: {exc}") from exc


def _validate_passthrough_base_url(cfg: RoutingConfig, *, source: str) -> None:
    """Reject `gateway: passthrough` rules that omit `base_url` (no upstream to forward to)."""
    for idx, rule in enumerate(cfg.rules):
        if rule.target.gateway == "passthrough" and not rule.target.base_url:
            label = format_rule_label(rule, idx)
            raise RoutingConfigError(
                f"{source}: {label}: gateway='passthrough' requires target.base_url"
            )


def _transforms_touch_body(transforms: Iterable[PreTransform]) -> bool:
    return any(isinstance(rw, (SetModel, JqPatch, Compress)) for rw in transforms)


def _pre_transforms_unconditionally_touch_body(entries: Iterable[PreTransform]) -> bool:
    """True iff a bare (non-guarded) body-touching transform sits in pre_transforms."""
    bare = [e for e in entries if not isinstance(e, GuardedTransforms)]
    return _transforms_touch_body(bare)


def _warn_passthrough_body_touch(cfg: RoutingConfig) -> None:
    """Debug-log passthrough rules that body-touching transforms would re-serialise."""
    pre_touches = _pre_transforms_unconditionally_touch_body(cfg.pre_transforms)
    for idx, rule in enumerate(cfg.rules):
        if rule.target.gateway != "passthrough":
            continue
        post_touches = _transforms_touch_body(rule.transforms)
        if pre_touches or post_touches:
            log.debug(
                "routing.passthrough_body_touch",
                rule=format_rule_label(rule, idx),
                pre_transforms_touch=pre_touches,
                post_transforms_touch=post_touches,
            )
