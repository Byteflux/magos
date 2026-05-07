"""``set_header``, ``add_header``, ``remove_header`` tests."""

from __future__ import annotations

from magos.routing import AddHeader, NamedValue, RemoveHeader, SetHeader
from magos.routing.rewrites import apply_transforms
from tests.routing._helpers import make_req

# --- SetHeader ---


def test_set_header_inserts() -> None:
    req = make_req()
    out = apply_transforms(req, [SetHeader(set_header=NamedValue(name="x-foo", value="bar"))])
    assert out.headers["x-foo"] == "bar"
    assert out.body_dirty is False


def test_set_header_overwrites() -> None:
    req = make_req(headers={"x-foo": "old"})
    out = apply_transforms(req, [SetHeader(set_header=NamedValue(name="x-foo", value="new"))])
    assert out.headers["x-foo"] == "new"


def test_set_header_lowercases_name() -> None:
    req = make_req(headers={"x-foo": "old"})
    out = apply_transforms(req, [SetHeader(set_header=NamedValue(name="X-Foo", value="new"))])
    assert out.headers == {"x-foo": "new"}


def test_set_header_idempotent_repeat() -> None:
    req = make_req()
    once = apply_transforms(req, [SetHeader(set_header=NamedValue(name="x-foo", value="bar"))])
    twice = apply_transforms(
        req,
        [
            SetHeader(set_header=NamedValue(name="x-foo", value="other")),
            SetHeader(set_header=NamedValue(name="x-foo", value="bar")),
        ],
    )
    assert once.headers == twice.headers


# --- AddHeader ---


def test_add_header_inserts_when_absent() -> None:
    req = make_req()
    out = apply_transforms(req, [AddHeader(add_header=NamedValue(name="x-foo", value="bar"))])
    assert out.headers["x-foo"] == "bar"


def test_add_header_no_op_when_present() -> None:
    req = make_req(headers={"x-foo": "existing"})
    out = apply_transforms(req, [AddHeader(add_header=NamedValue(name="x-foo", value="new"))])
    assert out.headers["x-foo"] == "existing"


def test_add_header_collision_is_case_insensitive() -> None:
    req = make_req(headers={"x-foo": "existing"})
    out = apply_transforms(req, [AddHeader(add_header=NamedValue(name="X-Foo", value="new"))])
    assert out.headers == {"x-foo": "existing"}


# --- RemoveHeader ---


def test_remove_header_drops_present() -> None:
    req = make_req(headers={"x-foo": "bar", "x-keep": "1"})
    out = apply_transforms(req, [RemoveHeader(remove_header="x-foo")])
    assert "x-foo" not in out.headers
    assert out.headers["x-keep"] == "1"


def test_remove_header_missing_is_noop() -> None:
    req = make_req(headers={"x-keep": "1"})
    out = apply_transforms(req, [RemoveHeader(remove_header="x-foo")])
    assert out.headers == {"x-keep": "1"}


def test_remove_header_lowercases_name() -> None:
    req = make_req(headers={"x-foo": "bar"})
    out = apply_transforms(req, [RemoveHeader(remove_header="X-Foo")])
    assert "x-foo" not in out.headers
