"""Thin ``jq`` wrapper: config-load validation + truthy predicate evaluation."""

from __future__ import annotations

import functools
from typing import Any

import jq


class JqCompileError(ValueError):
    """Raised when a jq expression fails to parse."""


# Compile-cache: avoids per-request jq.compile cost.
@functools.lru_cache(maxsize=256)
def _compile(expr: str) -> Any:
    return jq.compile(expr)


def check_program(expr: str) -> None:
    """Compile ``expr`` (via the cache); raise ``JqCompileError`` on parse failure."""
    try:
        _compile(expr)
    except Exception as exc:  # jq raises bare ValueError; surface uniformly.
        raise JqCompileError(f"invalid jq expression {expr!r}: {exc}") from exc


def evaluate_predicate(expr: str, value: Any) -> bool:
    """Run ``expr`` against ``value``; return Python-truthy on the first result."""
    result = _compile(expr).input_value(value).first()
    return bool(result)


def evaluate_patch(expr: str, value: Any) -> Any:
    """Run ``expr`` against ``value``; return the raw first result. Caller checks shape."""
    return _compile(expr).input_value(value).first()
