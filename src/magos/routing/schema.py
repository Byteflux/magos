"""Pydantic schemas for declarative routing config. See ``docs/routing/grammar.md``.

Union variants (``Matcher``, ``MatchExpr``, ``Rewrite``) use single-key +
``extra="forbid"`` so pydantic's smart-mode union dispatches by present key.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    """Frozen + extra-forbidding base for every routing schema."""

    # populate_by_name lets callers use ``Not(not_=...)`` since ``not`` is reserved.
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


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


CompressMode = Literal["token", "cache"]


class CompressOptions(_Frozen):
    """Compression knobs; see ``docs/headroom/pipeline.md``."""

    mode: CompressMode = "token"
    compress_user_messages: bool = False
    compress_system_messages: bool = True
    protect_recent: int = Field(default=4, ge=0)
    protect_analysis_context: bool = True
    target_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    min_tokens_to_compress: int = Field(default=250, ge=0)
    kompress_model: str | None = None
    model_limit: int | None = Field(default=None, ge=1024)
    """Context-window override; ``None`` auto-detects. See ``docs/headroom/model-limit.md``."""

    smart_routing: bool = True
    """When True, use ContentRouter for per-content-type dispatch (default).
    When False, use the legacy SmartCrusher-only path. See
    ``docs/headroom/pipeline.md``."""

    code_aware: bool = False
    """Enable AST-aware code compression in ContentRouter. Requires
    tree-sitter; ignored when ``smart_routing`` is False."""

    intelligent_context: bool = True
    """When True, use IntelligentContextManager (score-based fitting).
    When False, fall back to RollingWindow (last-N-turns)."""

    keep_last_turns: int = Field(default=4, ge=0)
    """Recent turns the context manager must preserve verbatim."""

    ccr_enabled: bool = True
    """When True (default), the compress rewrite injects ``headroom_retrieve``
    into the request whenever post-compression messages contain compression
    markers, and dispatch intercepts the model's tool calls. Set False to
    disable CCR for this rule (compression still runs; markers are emitted
    but no tool / instruction injection)."""

    ccr_inject_tool: bool = True
    """Inject the ``headroom_retrieve`` tool definition into ``body.tools``.
    Has effect only when ``ccr_enabled`` is True. Disable if a client
    distributes the tool via MCP (server-side) and re-injection would
    duplicate."""

    ccr_inject_instructions: bool = True
    """Inject system-message instructions describing how to call
    ``headroom_retrieve``. Has effect only when ``ccr_enabled`` is True
    and the prefix-cache freeze count is zero (otherwise instruction
    injection is skipped to preserve the cache, regardless of this flag)."""


class Compress(_Frozen):
    compress: CompressOptions = Field(default_factory=CompressOptions)


Rewrite = SetModel | SetHeader | RemoveHeader | AddHeader | JqPatch | Compress


DispatchMode = Literal["translate", "passthrough"]
AuthHeaderShape = Literal["bearer", "x-api-key"]


class Action(_Frozen):
    provider: str = Field(min_length=1)
    mode: DispatchMode
    base_url: str | None = None
    api_key_env: str | None = None
    auth_header: AuthHeaderShape | None = None
    """Auth-header shape override. See ``docs/routing/api-keys.md``."""


class Rule(_Frozen):
    name: str | None = None
    match: MatchExpr
    rewrites: list[Rewrite] = Field(default_factory=list)
    action: Action


class GuardedRewrites(_Frozen):
    """Pre-rewrite group gated by a match expression. See ``docs/routing/grammar.md``."""

    match: MatchExpr
    rewrites: list[Rewrite] = Field(min_length=1)


PreRewrite = Rewrite | GuardedRewrites


class RoutingConfig(_Frozen):
    pre_rewrites: list[PreRewrite] = Field(default_factory=list)
    rules: list[Rule] = Field(min_length=1)


def config_uses_compress(cfg: RoutingConfig) -> bool:
    """True iff any pre-rewrite or rule rewrite is a ``compress`` op.

    Walks into ``GuardedRewrites`` so a guarded pre-rewrite still
    counts; the engine evaluates them at request time.
    """
    for entry in cfg.pre_rewrites:
        if isinstance(entry, Compress):
            return True
        if isinstance(entry, GuardedRewrites) and any(
            isinstance(rw, Compress) for rw in entry.rewrites
        ):
            return True
    return any(isinstance(rw, Compress) for rule in cfg.rules for rw in rule.rewrites)
