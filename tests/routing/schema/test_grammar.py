"""Match grammar: matcher dispatch, atoms, combinators, alias for ``not``."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from magos.routing.schema import (
    AllOf,
    AnyOf,
    EndpointAtom,
    GlobMatcher,
    HeaderAtom,
    JqAtom,
    LiteralMatcher,
    ModelAtom,
    Not,
    RegexMatcher,
)

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
