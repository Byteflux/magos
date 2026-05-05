"""Thin ``jq`` wrapper: config-load validation + truthy predicate evaluation."""

from __future__ import annotations

from typing import Any

import jq


class JqCompileError(ValueError):
    """Raised when a jq expression fails to parse."""


def check_program(expr: str) -> None:
    """Compile ``expr``; raise ``JqCompileError`` on parse failure."""
    try:
        jq.compile(expr)
    except Exception as exc:  # jq raises bare ValueError; surface uniformly.
        raise JqCompileError(f"invalid jq expression {expr!r}: {exc}") from exc


def evaluate_predicate(expr: str, value: Any) -> bool:
    """Run ``expr`` against ``value``; return Python-truthy on the first result."""
    program = jq.compile(expr)
    result = program.input_value(value).first()
    return bool(result)


def evaluate_patch(expr: str, value: Any) -> Any:
    """Run ``expr`` against ``value``; return the raw first result. Caller checks shape."""
    program = jq.compile(expr)
    return program.input_value(value).first()
