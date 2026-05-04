"""``set_model`` rewrite tests."""

from __future__ import annotations

from magos.routing import SetModel
from magos.routing.rewrites import apply_rewrites
from tests.routing._helpers import make_req


def test_set_model_replaces_value_and_marks_dirty() -> None:
    req = make_req(body={"model": "old", "max_tokens": 8})
    out = apply_rewrites(req, [SetModel(set_model="new")])
    assert out.body["model"] == "new"
    assert out.body["max_tokens"] == 8
    assert out.body_dirty is True


def test_set_model_does_not_mutate_input() -> None:
    body = {"model": "old"}
    req = make_req(body=body)
    apply_rewrites(req, [SetModel(set_model="new")])
    assert body == {"model": "old"}
