"""Rewrite primitives: dispatch by key + ``CompressOptions`` knobs."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from magos.routing.schema import (
    AddHeader,
    CompressOptions,
    JqPatch,
    RemoveHeader,
    Rule,
    SetHeader,
    SetModel,
)

# --- Rewrite union dispatches by key ---


@pytest.mark.parametrize(
    ("payload", "cls"),
    [
        ({"set_model": "claude-haiku-4-5-20251001"}, SetModel),
        ({"set_header": {"name": "x-foo", "value": "bar"}}, SetHeader),
        ({"add_header": {"name": "x-foo", "value": "bar"}}, AddHeader),
        ({"remove_header": "x-foo"}, RemoveHeader),
        ({"jq_patch": '.messages[0].content = "x"'}, JqPatch),
    ],
)
def test_rewrite_dispatch(payload: dict[str, object], cls: type) -> None:
    rule = Rule.model_validate(
        {
            "match": {"endpoint": {"literal": "/v1/messages"}},
            "rewrites": [payload],
            "target": {"provider": "openai", "gateway": "translate"},
        }
    )
    assert isinstance(rule.rewrites[0], cls)


# --- CompressOptions pipeline-shape knobs ---


def test_compress_options_new_pipeline_fields_have_proxy_modern_defaults() -> None:
    opts = CompressOptions()
    assert opts.smart_routing is True
    assert opts.code_aware is False
    assert opts.intelligent_context is True
    assert opts.keep_last_turns == 4


def test_compress_options_accepts_legacy_shape() -> None:
    opts = CompressOptions(
        smart_routing=False,
        intelligent_context=False,
        keep_last_turns=8,
    )
    assert opts.smart_routing is False
    assert opts.intelligent_context is False
    assert opts.keep_last_turns == 8


def test_compress_options_rejects_negative_keep_last_turns() -> None:
    with pytest.raises(ValidationError):
        CompressOptions(keep_last_turns=-1)


def test_compress_options_ccr_defaults() -> None:
    opts = CompressOptions()
    assert opts.ccr_enabled is True
    assert opts.ccr_inject_tool is True
    assert opts.ccr_inject_instructions is True


def test_compress_options_ccr_can_be_disabled() -> None:
    opts = CompressOptions(ccr_enabled=False)
    assert opts.ccr_enabled is False


def test_compress_options_ccr_partial_opt_out() -> None:
    """ccr_enabled stays True but instructions can be turned off independently."""
    opts = CompressOptions(ccr_inject_instructions=False)
    assert opts.ccr_enabled is True
    assert opts.ccr_inject_tool is True
    assert opts.ccr_inject_instructions is False
