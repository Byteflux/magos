"""``jq_patch`` rewrite tests."""

from __future__ import annotations

import pytest

from magos.routing import JqPatch
from magos.routing.rewrites import RewriteError, apply_transforms
from tests.routing._helpers import make_req


def test_jq_patch_adds_field_and_marks_dirty() -> None:
    req = make_req(body={"model": "x"})
    out = apply_transforms(req, [JqPatch(jq_patch=".max_tokens //= 1024")])
    assert out.body["max_tokens"] == 1024
    assert out.body["model"] == "x"
    assert out.body_dirty is True


def test_jq_patch_overwrites_field() -> None:
    req = make_req(body={"model": "old"})
    out = apply_transforms(req, [JqPatch(jq_patch='.model = "new"')])
    assert out.body["model"] == "new"


def test_jq_patch_non_object_result_raises() -> None:
    req = make_req(body={"model": "x"})
    with pytest.raises(RewriteError, match="must be a JSON object"):
        apply_transforms(req, [JqPatch(jq_patch=".model")])


def test_jq_patch_null_result_raises() -> None:
    req = make_req(body={})
    with pytest.raises(RewriteError, match="NoneType"):
        apply_transforms(req, [JqPatch(jq_patch=".missing")])
