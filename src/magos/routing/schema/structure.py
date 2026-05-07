"""Top-level routing structure: targets, rules, guarded pre-rewrites, root config."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from ._base import _Frozen
from .grammar import MatchExpr
from .rewrites import Rewrite

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
    rewrites: list[Rewrite] = Field(default_factory=list)
    target: Target


class GuardedRewrites(_Frozen):
    """Pre-rewrite group gated by a match expression. See ``docs/routing/grammar.md``."""

    match: MatchExpr
    rewrites: list[Rewrite] = Field(min_length=1)


PreRewrite = Rewrite | GuardedRewrites


class RoutingConfig(_Frozen):
    pre_rewrites: list[PreRewrite] = Field(default_factory=list)
    rules: list[Rule] = Field(min_length=1)
