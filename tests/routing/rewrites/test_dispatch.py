"""Cross-primitive applicator behaviour: identity, body_dirty, sequencing, mutation."""

from __future__ import annotations

from magos.routing import (
    AddHeader,
    JqPatch,
    NamedValue,
    RemoveHeader,
    SetHeader,
    SetModel,
)
from magos.routing.rewrites import apply_rewrites
from tests.routing._helpers import make_req


def test_empty_rewrite_list_returns_input_identity() -> None:
    req = make_req(body={"model": "x"})
    assert apply_rewrites(req, []) is req


def test_header_only_ops_preserve_body_dirty_false() -> None:
    req = make_req(headers={"x-old": "1"})
    out = apply_rewrites(
        req,
        [
            SetHeader(set_header=NamedValue(name="x-new", value="2")),
            RemoveHeader(remove_header="x-old"),
            AddHeader(add_header=NamedValue(name="x-other", value="3")),
        ],
    )
    assert out.body_dirty is False


def test_body_dirty_persists_once_set() -> None:
    req = make_req(body={"model": "x"})
    out = apply_rewrites(
        req,
        [
            SetModel(set_model="y"),
            SetHeader(set_header=NamedValue(name="x-foo", value="bar")),
        ],
    )
    assert out.body_dirty is True


def test_sequential_rewrites_apply_in_order() -> None:
    req = make_req(body={"model": "a"})
    out = apply_rewrites(
        req,
        [
            SetModel(set_model="b"),
            JqPatch(jq_patch='.model = "c"'),
        ],
    )
    assert out.body["model"] == "c"


def test_input_request_is_not_mutated() -> None:
    headers = {"x-foo": "bar"}
    body = {"model": "old", "max_tokens": 8}
    req = make_req(headers=headers, body=body)
    apply_rewrites(
        req,
        [
            SetModel(set_model="new"),
            SetHeader(set_header=NamedValue(name="x-foo", value="overwritten")),
            RemoveHeader(remove_header="x-foo"),
            JqPatch(jq_patch=".max_tokens = 16"),
        ],
    )
    assert headers == {"x-foo": "bar"}
    assert body == {"model": "old", "max_tokens": 8}
