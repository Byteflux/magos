"""``jq_patch`` rewrite: arbitrary jq program over the body.

The jq program must return a JSON object — anything else is a config
error and surfaces as ``RewriteError`` (caught by ``route()`` and
turned into a 503 ``dispatch_error`` envelope). Flips ``body_dirty``
so passthrough's byte-exact path is bypassed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from magos.routing.jq_compat import evaluate_patch
from magos.routing.request import RoutedRequest
from magos.routing.schema import JqPatch


class RewriteError(ValueError):
    """Raised when a rewrite cannot be applied (e.g., jq_patch shape error)."""


def apply_jq_patch(req: RoutedRequest, rw: JqPatch) -> RoutedRequest:
    result: Any = evaluate_patch(rw.jq_patch, dict(req.body))
    if not isinstance(result, Mapping):
        raise RewriteError(
            f"jq_patch result must be a JSON object, got {type(result).__name__}: {rw.jq_patch!r}"
        )
    return replace(req, body=dict(result), body_dirty=True)
