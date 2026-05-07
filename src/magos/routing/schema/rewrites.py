"""Rewrite primitives: header / model / jq / compress operations.

Every primitive is a single-key ``_Frozen`` model so the ``Rewrite``
union dispatches by present key in pydantic smart mode.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from ._base import _Frozen


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
