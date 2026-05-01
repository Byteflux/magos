"""Thin wrapper around the ``jq`` PyPI package.

Two responsibilities:

- ``check_program``: parse-validate a jq expression at config-load time so
  syntax errors surface with file + rule context instead of at first request.
- ``Truthy`` evaluation: the predicate semantics for match atoms — a jq
  expression matches iff its first result value is truthy under Python rules
  (``None``, ``False``, ``0``, ``""``, empty containers all falsy).

Compiled programs are not cached here; the engine layer keeps a per-rule
cache keyed by rule identity (Slice 4) so we re-compile on demand instead of
fighting pydantic's frozen models with private attrs.
"""

from __future__ import annotations

from typing import Any

import jq


class JqCompileError(ValueError):
    """Raised when a jq expression fails to parse."""


def check_program(expr: str) -> None:
    """Compile ``expr`` and discard the result; raises on parse failure."""
    try:
        jq.compile(expr)
    except Exception as exc:  # jq raises a bare ValueError; surface uniformly.
        raise JqCompileError(f"invalid jq expression {expr!r}: {exc}") from exc


def evaluate_predicate(expr: str, value: Any) -> bool:
    """Run ``expr`` against ``value``; return Python-truthy on the first result."""
    program = jq.compile(expr)
    result = program.input_value(value).first()
    return bool(result)


def evaluate_patch(expr: str, value: Any) -> Any:
    """Run ``expr`` against ``value``; return the first result.

    Caller is responsible for validating the result shape (``jq_patch``
    requires a JSON object). This helper stays shape-agnostic so it can be
    reused for predicates that need the raw result.
    """
    program = jq.compile(expr)
    return program.input_value(value).first()
