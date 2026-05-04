"""``set_model`` rewrite: replace the body's ``model`` field.

Flips ``body_dirty=True`` so the dispatcher re-serialises the body
before forwarding (passthrough's byte-exact path is no longer valid
once the body has been edited).
"""

from __future__ import annotations

from dataclasses import replace

from magos.routing.request import RoutedRequest
from magos.routing.schema import SetModel


def apply_set_model(req: RoutedRequest, rw: SetModel) -> RoutedRequest:
    new_body = dict(req.body)
    new_body["model"] = rw.set_model
    return replace(req, body=new_body, body_dirty=True)
