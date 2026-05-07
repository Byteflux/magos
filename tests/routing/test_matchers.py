"""Tests for the pure match-expression evaluator."""

from __future__ import annotations

import pytest

from magos.routing import (
    AllOf,
    AnyOf,
    EndpointAtom,
    GlobMatcher,
    HeaderAtom,
    HeaderPair,
    JqAtom,
    LiteralMatcher,
    ModelAtom,
    Not,
    RegexMatcher,
)
from magos.routing.match import matches
from tests.routing._helpers import make_req as _req

# --- ModelAtom ---


def test_model_literal_matches_exact() -> None:
    expr = ModelAtom(model=LiteralMatcher(literal="claude-3"))
    assert matches(expr, _req(body={"model": "claude-3"}))


def test_model_literal_rejects_different() -> None:
    expr = ModelAtom(model=LiteralMatcher(literal="claude-3"))
    assert not matches(expr, _req(body={"model": "claude-4"}))


def test_model_glob_matches_wildcard() -> None:
    expr = ModelAtom(model=GlobMatcher(glob="claude-*"))
    assert matches(expr, _req(body={"model": "claude-3"}))
    assert matches(expr, _req(body={"model": "claude-haiku-4-5-20251001"}))
    assert not matches(expr, _req(body={"model": "gpt-4"}))


def test_model_glob_is_case_sensitive() -> None:
    expr = ModelAtom(model=GlobMatcher(glob="Claude-*"))
    assert not matches(expr, _req(body={"model": "claude-3"}))


def test_model_regex_uses_fullmatch() -> None:
    expr = ModelAtom(model=RegexMatcher(regex="claude-"))
    # fullmatch: regex must match the whole string, not just a prefix.
    assert not matches(expr, _req(body={"model": "claude-3"}))
    assert matches(expr, _req(body={"model": "claude-"}))


def test_model_regex_explicit_anchors() -> None:
    expr = ModelAtom(model=RegexMatcher(regex=r"claude-\d+"))
    assert matches(expr, _req(body={"model": "claude-3"}))
    assert not matches(expr, _req(body={"model": "claude-x"}))


def test_model_missing_field_does_not_match_any_literal() -> None:
    expr = ModelAtom(model=LiteralMatcher(literal="x"))
    # Missing body.model defaults to "" which does not match the literal "x".
    assert not matches(expr, _req(body={}))


def test_model_missing_field_matches_empty_regex() -> None:
    # Empty-string regex confirms the default-to-empty-string contract.
    expr = ModelAtom(model=RegexMatcher(regex="^$"))
    assert matches(expr, _req(body={}))


# --- EndpointAtom ---


@pytest.mark.parametrize(
    "endpoint",
    ["/v1/messages", "/v1/messages/count_tokens", "/v1/chat/completions"],
)
def test_endpoint_literal(endpoint: str) -> None:
    expr = EndpointAtom(endpoint=LiteralMatcher(literal=endpoint))
    assert matches(expr, _req(endpoint=endpoint))  # type: ignore[arg-type]


def test_endpoint_glob() -> None:
    expr = EndpointAtom(endpoint=GlobMatcher(glob="/v1/messages*"))
    assert matches(expr, _req(endpoint="/v1/messages"))
    assert matches(expr, _req(endpoint="/v1/messages/count_tokens"))
    assert not matches(expr, _req(endpoint="/v1/chat/completions"))


# --- HeaderAtom (existential) ---


def test_header_existential_matches_any_pair() -> None:
    expr = HeaderAtom(
        header=HeaderPair(
            name=LiteralMatcher(literal="x-route"),
            value=LiteralMatcher(literal="anthropic"),
        )
    )
    headers = {"x-route": "anthropic", "x-other": "value"}
    assert matches(expr, _req(headers=headers))


def test_header_requires_both_name_and_value_match_on_same_pair() -> None:
    expr = HeaderAtom(
        header=HeaderPair(
            name=LiteralMatcher(literal="x-foo"),
            value=LiteralMatcher(literal="bar"),
        )
    )
    headers = {"x-foo": "other", "x-baz": "bar"}
    assert not matches(expr, _req(headers=headers))


def test_header_glob_value_matches_bearer() -> None:
    expr = HeaderAtom(
        header=HeaderPair(
            name=LiteralMatcher(literal="authorization"),
            value=GlobMatcher(glob="Bearer *"),
        )
    )
    assert matches(expr, _req(headers={"authorization": "Bearer abc123"}))
    assert not matches(expr, _req(headers={"authorization": "Basic xxx"}))


def test_header_regex_name() -> None:
    expr = HeaderAtom(
        header=HeaderPair(
            name=RegexMatcher(regex=r"x-magos-.*"),
            value=LiteralMatcher(literal="1"),
        )
    )
    assert matches(expr, _req(headers={"x-magos-tier": "1"}))


def test_header_no_match_on_empty_headers() -> None:
    expr = HeaderAtom(
        header=HeaderPair(
            name=LiteralMatcher(literal="x-foo"),
            value=LiteralMatcher(literal="bar"),
        )
    )
    assert not matches(expr, _req(headers={}))


# --- JqAtom (truthy predicate) ---


def test_jq_truthy_matches() -> None:
    expr = JqAtom(jq=".stream == true")
    assert matches(expr, _req(body={"stream": True}))


def test_jq_falsy_does_not_match() -> None:
    expr = JqAtom(jq=".stream == true")
    assert not matches(expr, _req(body={"stream": False}))
    assert not matches(expr, _req(body={}))


def test_jq_zero_is_falsy() -> None:
    expr = JqAtom(jq=".count")
    assert not matches(expr, _req(body={"count": 0}))


def test_jq_empty_array_is_falsy() -> None:
    # Python truthiness: empty list -> False.
    expr = JqAtom(jq=".tags")
    assert not matches(expr, _req(body={"tags": []}))


def test_jq_non_empty_array_is_truthy() -> None:
    expr = JqAtom(jq=".tools")
    assert matches(expr, _req(body={"tools": [{"name": "x"}]}))


def test_jq_predicate_with_length_check() -> None:
    expr = JqAtom(jq=".tools | length > 0")
    assert matches(expr, _req(body={"tools": [{"name": "x"}]}))
    assert not matches(expr, _req(body={"tools": []}))


# --- Combinators ---


def test_all_of_requires_every_child() -> None:
    expr = AllOf(
        all_of=[
            EndpointAtom(endpoint=LiteralMatcher(literal="/v1/messages")),
            ModelAtom(model=GlobMatcher(glob="claude-*")),
        ]
    )
    assert matches(expr, _req(endpoint="/v1/messages", body={"model": "claude-3"}))
    assert not matches(expr, _req(endpoint="/v1/messages", body={"model": "gpt-4"}))


def test_any_of_succeeds_on_first_match() -> None:
    expr = AnyOf(
        any_of=[
            ModelAtom(model=LiteralMatcher(literal="gpt-4")),
            ModelAtom(model=GlobMatcher(glob="claude-*")),
        ]
    )
    assert matches(expr, _req(body={"model": "claude-3"}))
    assert matches(expr, _req(body={"model": "gpt-4"}))
    assert not matches(expr, _req(body={"model": "mistral-7b"}))


def test_not_inverts() -> None:
    # Constructed via model_validate because the field alias "not" collides
    # with the Python keyword and mypy's pydantic plugin does not recognise
    # the field-name kwarg even with populate_by_name=True.
    expr = Not.model_validate({"not": {"model": {"glob": "claude-*"}}})
    assert matches(expr, _req(body={"model": "gpt-4"}))
    assert not matches(expr, _req(body={"model": "claude-3"}))


def test_deeply_nested_recursion() -> None:
    # See test_not_inverts re: model_validate for Not.
    expr = AllOf.model_validate(
        {
            "all_of": [
                {"endpoint": {"literal": "/v1/messages"}},
                {
                    "any_of": [
                        {"model": {"glob": "claude-*"}},
                        {
                            "all_of": [
                                {"model": {"glob": "gpt-*"}},
                                {"not": {"jq": ".stream == true"}},
                            ]
                        },
                    ]
                },
            ]
        }
    )
    # gpt-4 + non-streaming -> deep AllOf branch matches.
    assert matches(expr, _req(endpoint="/v1/messages", body={"model": "gpt-4", "stream": False}))
    # gpt-4 + streaming -> Not(streaming) fails, AnyOf fails, AllOf fails.
    assert not matches(expr, _req(endpoint="/v1/messages", body={"model": "gpt-4", "stream": True}))
    # claude-3 -> first AnyOf branch wins regardless of stream.
    assert matches(expr, _req(endpoint="/v1/messages", body={"model": "claude-3", "stream": True}))
