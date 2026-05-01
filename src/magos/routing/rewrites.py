"""Pure mutators for the routing pipeline.

Each rewrite consumes a ``RoutedRequest`` and returns a new one. The frozen
dataclass forbids in-place mutation, so we copy ``headers`` and ``body``
defensively and use ``dataclasses.replace`` to produce successors. Body-
touching ops (``SetModel``, ``JqPatch``) flip ``body_dirty`` so the
dispatcher knows it must re-serialise instead of forwarding ``raw_body``
verbatim under passthrough.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from magos.routing.jq_compat import evaluate_patch
from magos.routing.models import (
    AddHeader,
    JqPatch,
    RemoveHeader,
    Rewrite,
    SetHeader,
    SetModel,
)
from magos.routing.request import RoutedRequest


class RewriteError(ValueError):
    """Raised when a rewrite cannot be applied (e.g., jq_patch shape error)."""


def apply_rewrites(req: RoutedRequest, rewrites: Sequence[Rewrite]) -> RoutedRequest:
    """Apply ``rewrites`` in list order; return a new RoutedRequest.

    Empty list returns ``req`` unchanged (same identity). Original headers
    and body are never mutated.
    """
    if not rewrites:
        return req
    out = req
    for rw in rewrites:
        out = _apply_one(out, rw)
    return out


def _apply_one(req: RoutedRequest, rw: Rewrite) -> RoutedRequest:  # noqa: PLR0911
    if isinstance(rw, SetModel):
        new_body = dict(req.body)
        new_body["model"] = rw.set_model
        return replace(req, body=new_body, body_dirty=True)
    if isinstance(rw, SetHeader):
        return replace(
            req, headers=_with_header(req.headers, rw.set_header.name, rw.set_header.value)
        )
    if isinstance(rw, AddHeader):
        key = rw.add_header.name.lower()
        if key in req.headers:
            return req
        return replace(
            req, headers=_with_header(req.headers, rw.add_header.name, rw.add_header.value)
        )
    if isinstance(rw, RemoveHeader):
        key = rw.remove_header.lower()
        if key not in req.headers:
            return req
        new_headers = dict(req.headers)
        del new_headers[key]
        return replace(req, headers=new_headers)
    if isinstance(rw, JqPatch):
        result: Any = evaluate_patch(rw.jq_patch, dict(req.body))
        if not isinstance(result, Mapping):
            raise RewriteError(
                f"jq_patch result must be a JSON object, got "
                f"{type(result).__name__}: {rw.jq_patch!r}"
            )
        return replace(req, body=dict(result), body_dirty=True)
    raise TypeError(f"unhandled Rewrite variant: {type(rw).__name__}")


def _with_header(headers: Mapping[str, str], name: str, value: str) -> dict[str, str]:
    new_headers = dict(headers)
    new_headers[name.lower()] = value
    return new_headers
