"""Top-level routing structure: targets, rules, guarded pre-transforms, root config."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import _Frozen
from .grammar import MatchExpr
from .rewrites import AddHeader, Compress, JqPatch, RemoveHeader, SetHeader, SetModel

GatewayMode = Literal["translate", "passthrough"]
AuthHeaderShape = Literal["bearer", "x-api-key"]


class Target(_Frozen):
    provider: str = Field(min_length=1)
    gateway: GatewayMode
    base_url: str | None = None
    api_key_env: str | None = None
    auth_header: AuthHeaderShape | None = None
    """Auth-header shape override. See ``docs/routing/api-keys.md``."""


class Rule(_Frozen):
    name: str | None = None
    match: MatchExpr
    transforms: list[SetModel | SetHeader | RemoveHeader | AddHeader | JqPatch | Compress] = Field(
        default_factory=list
    )
    target: Target


class GuardedTransforms(_Frozen):
    """Pre-transform group gated by a match expression. See ``docs/routing/grammar.md``."""

    match: MatchExpr
    transforms: list[SetModel | SetHeader | RemoveHeader | AddHeader | JqPatch | Compress] = Field(
        min_length=1
    )


PreTransform = (
    SetModel | SetHeader | RemoveHeader | AddHeader | JqPatch | Compress | GuardedTransforms
)


class RoutingConfig(_Frozen):
    pre_transforms: list[
        SetModel | SetHeader | RemoveHeader | AddHeader | JqPatch | Compress | GuardedTransforms
    ] = Field(default_factory=list)
    rules: list[Rule] = Field(min_length=1)
