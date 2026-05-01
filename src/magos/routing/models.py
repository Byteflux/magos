"""Pydantic schemas for declarative routing config.

Mirrors the YAML grammar in ``magos.yaml``:

- ``RoutingConfig`` has optional global ``pre_rewrites`` and an ordered list
  of ``rules``.
- ``Rule`` has a ``match`` expression, optional per-rule ``rewrites``
  (post-match), and an ``action``.
- ``MatchExpr`` is a recursive logical expression: combinators
  (``all_of`` / ``any_of`` / ``not``) plus atoms (``model`` / ``header`` /
  ``endpoint`` / ``jq``).
- Atoms use a tagged ``Matcher`` (``literal`` / ``glob`` / ``regex``)
  except ``jq`` which is a free-form expression.

Variants of the unions ``Matcher``, ``MatchExpr``, ``Rewrite`` are single-key
+ ``extra="forbid"``; pydantic's smart-mode union dispatches on the present
key without an explicit discriminator field.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    """Frozen + extra-forbidding base, shared by every routing schema."""

    model_config = ConfigDict(frozen=True, extra="forbid")


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


class AllOf(_Frozen):
    all_of: list[MatchExpr] = Field(min_length=1)


class AnyOf(_Frozen):
    any_of: list[MatchExpr] = Field(min_length=1)


class Not(_Frozen):
    not_: MatchExpr = Field(alias="not")


# Single-key + extra="forbid" lets pydantic dispatch by which key is present.
MatchExpr = ModelAtom | HeaderAtom | EndpointAtom | JqAtom | AllOf | AnyOf | Not


AllOf.model_rebuild()
AnyOf.model_rebuild()
Not.model_rebuild()


class NamedValue(_Frozen):
    name: str = Field(min_length=1)
    value: str


class SetModel(_Frozen):
    set_model: str = Field(min_length=1)


class SetHeader(_Frozen):
    set_header: NamedValue


class RemoveHeader(_Frozen):
    remove_header: str = Field(min_length=1)


class AddHeader(_Frozen):
    add_header: NamedValue


class JqPatch(_Frozen):
    jq_patch: str = Field(min_length=1)


Rewrite = SetModel | SetHeader | RemoveHeader | AddHeader | JqPatch


DispatchMode = Literal["translate", "passthrough"]
CountTokensMode = Literal["local", "passthrough"]


class Action(_Frozen):
    provider: str = Field(min_length=1)
    mode: DispatchMode
    base_url: str | None = None
    api_key_env: str | None = None
    count_tokens_mode: CountTokensMode = "local"


class Rule(_Frozen):
    name: str | None = None
    match: MatchExpr
    rewrites: list[Rewrite] = Field(default_factory=list)
    action: Action


class RoutingConfig(_Frozen):
    pre_rewrites: list[Rewrite] = Field(default_factory=list)
    rules: list[Rule] = Field(min_length=1)
