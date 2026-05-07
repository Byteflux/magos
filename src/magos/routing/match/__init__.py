"""``magos.routing.match``: match-expression evaluation.

The evaluator walks the closed ``MatchExpr`` tree (atoms + AND/OR/NOT)
defined in :mod:`magos.routing.schema.grammar`. Public surface is the
:func:`matches` predicate.
"""

from __future__ import annotations

from .evaluator import matches

__all__ = ["matches"]
