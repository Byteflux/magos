"""Tests for the pure rewrite applicator."""

from __future__ import annotations

from typing import Any

import pytest

from magos.routing import (
    AddHeader,
    JqPatch,
    NamedValue,
    RemoveHeader,
    RoutedRequest,
    SetHeader,
    SetModel,
)
from magos.routing.rewrites import RewriteError, apply_rewrites


def _req(
    *,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    raw: bytes = b"",
    body_dirty: bool = False,
) -> RoutedRequest:
    return RoutedRequest(
        endpoint="/v1/messages",
        headers=headers or {},
        body=body or {},
        raw_body=raw,
        body_dirty=body_dirty,
    )


# --- Identity / no-op ---


def test_empty_rewrite_list_returns_input_identity() -> None:
    req = _req(body={"model": "x"})
    assert apply_rewrites(req, []) is req


# --- SetModel ---


def test_set_model_replaces_value_and_marks_dirty() -> None:
    req = _req(body={"model": "old", "max_tokens": 8})
    out = apply_rewrites(req, [SetModel(set_model="new")])
    assert out.body["model"] == "new"
    assert out.body["max_tokens"] == 8
    assert out.body_dirty is True


def test_set_model_does_not_mutate_input() -> None:
    body = {"model": "old"}
    req = _req(body=body)
    apply_rewrites(req, [SetModel(set_model="new")])
    assert body == {"model": "old"}


# --- SetHeader ---


def test_set_header_inserts() -> None:
    req = _req()
    out = apply_rewrites(req, [SetHeader(set_header=NamedValue(name="x-foo", value="bar"))])
    assert out.headers["x-foo"] == "bar"
    assert out.body_dirty is False


def test_set_header_overwrites() -> None:
    req = _req(headers={"x-foo": "old"})
    out = apply_rewrites(req, [SetHeader(set_header=NamedValue(name="x-foo", value="new"))])
    assert out.headers["x-foo"] == "new"


def test_set_header_lowercases_name() -> None:
    req = _req(headers={"x-foo": "old"})
    out = apply_rewrites(req, [SetHeader(set_header=NamedValue(name="X-Foo", value="new"))])
    assert out.headers == {"x-foo": "new"}


def test_set_header_idempotent_repeat() -> None:
    req = _req()
    once = apply_rewrites(req, [SetHeader(set_header=NamedValue(name="x-foo", value="bar"))])
    twice = apply_rewrites(
        req,
        [
            SetHeader(set_header=NamedValue(name="x-foo", value="other")),
            SetHeader(set_header=NamedValue(name="x-foo", value="bar")),
        ],
    )
    assert once.headers == twice.headers


# --- AddHeader ---


def test_add_header_inserts_when_absent() -> None:
    req = _req()
    out = apply_rewrites(req, [AddHeader(add_header=NamedValue(name="x-foo", value="bar"))])
    assert out.headers["x-foo"] == "bar"


def test_add_header_no_op_when_present() -> None:
    req = _req(headers={"x-foo": "existing"})
    out = apply_rewrites(req, [AddHeader(add_header=NamedValue(name="x-foo", value="new"))])
    assert out.headers["x-foo"] == "existing"


def test_add_header_collision_is_case_insensitive() -> None:
    req = _req(headers={"x-foo": "existing"})
    out = apply_rewrites(req, [AddHeader(add_header=NamedValue(name="X-Foo", value="new"))])
    assert out.headers == {"x-foo": "existing"}


# --- RemoveHeader ---


def test_remove_header_drops_present() -> None:
    req = _req(headers={"x-foo": "bar", "x-keep": "1"})
    out = apply_rewrites(req, [RemoveHeader(remove_header="x-foo")])
    assert "x-foo" not in out.headers
    assert out.headers["x-keep"] == "1"


def test_remove_header_missing_is_noop() -> None:
    req = _req(headers={"x-keep": "1"})
    out = apply_rewrites(req, [RemoveHeader(remove_header="x-foo")])
    assert out.headers == {"x-keep": "1"}


def test_remove_header_lowercases_name() -> None:
    req = _req(headers={"x-foo": "bar"})
    out = apply_rewrites(req, [RemoveHeader(remove_header="X-Foo")])
    assert "x-foo" not in out.headers


# --- JqPatch ---


def test_jq_patch_adds_field_and_marks_dirty() -> None:
    req = _req(body={"model": "x"})
    out = apply_rewrites(req, [JqPatch(jq_patch=".max_tokens //= 1024")])
    assert out.body["max_tokens"] == 1024
    assert out.body["model"] == "x"
    assert out.body_dirty is True


def test_jq_patch_overwrites_field() -> None:
    req = _req(body={"model": "old"})
    out = apply_rewrites(req, [JqPatch(jq_patch='.model = "new"')])
    assert out.body["model"] == "new"


def test_jq_patch_non_object_result_raises() -> None:
    req = _req(body={"model": "x"})
    with pytest.raises(RewriteError, match="must be a JSON object"):
        apply_rewrites(req, [JqPatch(jq_patch=".model")])


def test_jq_patch_null_result_raises() -> None:
    req = _req(body={})
    with pytest.raises(RewriteError, match="NoneType"):
        apply_rewrites(req, [JqPatch(jq_patch=".missing")])


# --- Header-only ops do not flip body_dirty ---


def test_header_only_ops_preserve_body_dirty_false() -> None:
    req = _req(headers={"x-old": "1"})
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
    req = _req(body={"model": "x"})
    out = apply_rewrites(
        req,
        [
            SetModel(set_model="y"),
            SetHeader(set_header=NamedValue(name="x-foo", value="bar")),
        ],
    )
    assert out.body_dirty is True


# --- Sequencing ---


def test_sequential_rewrites_apply_in_order() -> None:
    req = _req(body={"model": "a"})
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
    req = _req(headers=headers, body=body)
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
