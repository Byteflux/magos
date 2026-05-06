"""Schema tests for ``magos.routing.schema``.

Covers: every variant of every discriminated union, recursion through
combinators, ``extra="forbid"`` rejection, frozen-immutability, and the
required-field surface.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from magos.routing.schema import (
    Action,
    AddHeader,
    AllOf,
    AnyOf,
    CompressOptions,
    EndpointAtom,
    GlobMatcher,
    HeaderAtom,
    JqAtom,
    JqPatch,
    LiteralMatcher,
    ModelAtom,
    Not,
    RegexMatcher,
    RemoveHeader,
    RoutingConfig,
    Rule,
    SetHeader,
    SetModel,
)


def _action() -> Action:
    return Action.model_validate({"provider": "openai", "mode": "translate"})


# --- Matcher union dispatches by key ---


@pytest.mark.parametrize(
    ("payload", "cls"),
    [
        ({"literal": "x"}, LiteralMatcher),
        ({"glob": "x*"}, GlobMatcher),
        ({"regex": "^x"}, RegexMatcher),
    ],
)
def test_matcher_dispatch(payload: dict[str, str], cls: type) -> None:
    atom = ModelAtom.model_validate({"model": payload})
    assert isinstance(atom.model, cls)


def test_matcher_rejects_multiple_keys() -> None:
    with pytest.raises(ValidationError):
        ModelAtom.model_validate({"model": {"literal": "x", "regex": "^x"}})


def test_matcher_rejects_empty_string() -> None:
    with pytest.raises(ValidationError):
        ModelAtom.model_validate({"model": {"literal": ""}})


# --- Match expression union dispatches by key ---


def test_endpoint_atom_parses() -> None:
    expr = EndpointAtom.model_validate({"endpoint": {"literal": "/v1/messages"}})
    assert isinstance(expr.endpoint, LiteralMatcher)


def test_header_atom_parses_with_nested_matchers() -> None:
    expr = HeaderAtom.model_validate(
        {"header": {"name": {"literal": "x-foo"}, "value": {"glob": "Bearer *"}}}
    )
    assert isinstance(expr.header.name, LiteralMatcher)
    assert isinstance(expr.header.value, GlobMatcher)


def test_jq_atom_keeps_expression() -> None:
    atom = JqAtom.model_validate({"jq": ".stream == true"})
    assert atom.jq == ".stream == true"


# --- Combinators recurse ---


def test_allof_recursion() -> None:
    expr = AllOf.model_validate(
        {
            "all_of": [
                {"model": {"literal": "claude"}},
                {
                    "any_of": [
                        {"endpoint": {"literal": "/v1/messages"}},
                        {"jq": ".stream"},
                    ]
                },
            ]
        }
    )
    assert isinstance(expr.all_of[0], ModelAtom)
    assert isinstance(expr.all_of[1], AnyOf)
    assert isinstance(expr.all_of[1].any_of[0], EndpointAtom)
    assert isinstance(expr.all_of[1].any_of[1], JqAtom)


def test_not_uses_yaml_alias() -> None:
    expr = Not.model_validate({"not": {"endpoint": {"literal": "/v1/messages"}}})
    assert isinstance(expr.not_, EndpointAtom)


def test_combinators_reject_empty_lists() -> None:
    with pytest.raises(ValidationError):
        AllOf.model_validate({"all_of": []})
    with pytest.raises(ValidationError):
        AnyOf.model_validate({"any_of": []})


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
            "action": {"provider": "openai", "mode": "translate"},
        }
    )
    assert isinstance(rule.rewrites[0], cls)


# --- Action constraints ---


def test_action_rejects_unknown_mode() -> None:
    with pytest.raises(ValidationError):
        Action.model_validate({"provider": "openai", "mode": "swerve"})


def test_action_rejects_blank_provider() -> None:
    with pytest.raises(ValidationError):
        Action.model_validate({"provider": "", "mode": "translate"})


# --- extra="forbid" + frozen ---


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Rule.model_validate(
            {
                "match": {"endpoint": {"literal": "/v1/messages"}},
                "action": {"provider": "openai", "mode": "translate"},
                "unexpected": True,
            }
        )


def test_frozen_models_cannot_mutate() -> None:
    rule = Rule.model_validate(
        {
            "match": {"endpoint": {"literal": "/v1/messages"}},
            "action": {"provider": "openai", "mode": "translate"},
        }
    )
    with pytest.raises(ValidationError):
        rule.action.provider = "anthropic"


# --- Top-level invariants ---


def test_routing_config_requires_at_least_one_rule() -> None:
    with pytest.raises(ValidationError):
        RoutingConfig.model_validate({"rules": []})


def test_routing_config_round_trips() -> None:
    cfg = RoutingConfig.model_validate(
        {
            "pre_rewrites": [{"set_header": {"name": "x-magos", "value": "1"}}],
            "rules": [
                {
                    "name": "default",
                    "match": {"endpoint": {"literal": "/v1/messages"}},
                    "action": {"provider": "openai", "mode": "translate"},
                }
            ],
        }
    )
    assert cfg.rules[0].name == "default"
    assert isinstance(cfg.pre_rewrites[0], SetHeader)


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
