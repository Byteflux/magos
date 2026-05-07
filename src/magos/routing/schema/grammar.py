"""Match grammar: matchers (literal/glob/regex), atoms (model/header/
endpoint/jq/model_field), and combinators (all_of/any_of/not).

Union variants use single-key + ``extra="forbid"`` so pydantic's
smart-mode union dispatches by present key.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from magos.routing.schema.base import _Frozen


class LiteralMatcher(_Frozen):
    literal: str = Field(min_length=1)


class GlobMatcher(_Frozen):
    glob: str = Field(min_length=1)


class RegexMatcher(_Frozen):
    regex: str = Field(min_length=1)


Matcher = LiteralMatcher | GlobMatcher | RegexMatcher


class ModelAtom(_Frozen):
    model: Matcher


class HeaderPair(_Frozen):
    name: Matcher
    value: Matcher


class HeaderAtom(_Frozen):
    header: HeaderPair


class EndpointAtom(_Frozen):
    endpoint: Matcher


class JqAtom(_Frozen):
    jq: str = Field(min_length=1)


ModelFieldOp = Literal["eq", "gt", "gte", "lt", "lte", "contains", "in"]


class ModelFieldExpr(_Frozen):
    """Registry model-field comparison. See ``docs/registry/matchers.md``."""

    field: Literal[
        "context_size",
        "max_output",
        "input_cost",
        "output_cost",
        "cache_read_cost",
        "cache_write_cost",
        "input_modalities",
        "output_modalities",
    ]
    op: ModelFieldOp
    value: int | float | str | list[int | float | str]


class ModelFieldAtom(_Frozen):
    model_field: ModelFieldExpr


class AllOf(_Frozen):
    all_of: list[MatchExpr] = Field(min_length=1)


class AnyOf(_Frozen):
    any_of: list[MatchExpr] = Field(min_length=1)


class Not(_Frozen):
    not_: MatchExpr = Field(alias="not")


MatchExpr = ModelAtom | HeaderAtom | EndpointAtom | JqAtom | ModelFieldAtom | AllOf | AnyOf | Not


AllOf.model_rebuild()
AnyOf.model_rebuild()
Not.model_rebuild()
