"""Header mutators: ``set_header``, ``add_header``, ``remove_header``.

All three operate on a case-folded copy of ``req.headers``; magos
normalises inbound header keys to lowercase before they reach this
layer, so storing as lowercase keeps the dict consistent. Body is
untouched, so ``body_dirty`` does not flip.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from magos.routing.request import RoutedRequest
from magos.routing.schema import AddHeader, RemoveHeader, SetHeader


def apply_set_header(req: RoutedRequest, rw: SetHeader) -> RoutedRequest:
    return replace(req, headers=_with_header(req.headers, rw.set_header.name, rw.set_header.value))


def apply_add_header(req: RoutedRequest, rw: AddHeader) -> RoutedRequest:
    """No-op when the header is already present (set-if-absent semantics)."""
    key = rw.add_header.name.lower()
    if key in req.headers:
        return req
    return replace(req, headers=_with_header(req.headers, rw.add_header.name, rw.add_header.value))


def apply_remove_header(req: RoutedRequest, rw: RemoveHeader) -> RoutedRequest:
    key = rw.remove_header.lower()
    if key not in req.headers:
        return req
    new_headers = dict(req.headers)
    del new_headers[key]
    return replace(req, headers=new_headers)


def _with_header(headers: Mapping[str, str], name: str, value: str) -> dict[str, str]:
    new_headers = dict(headers)
    new_headers[name.lower()] = value
    return new_headers
