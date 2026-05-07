"""Transform primitives: header / model / jq / compress operations.

Every primitive is a single-key ``_Frozen`` model dispatched by present
key in pydantic smart mode.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from magos.routing.jq_compat import evaluate_patch
from magos.routing.rewrites.base import Rewriter

from ._base import _Frozen

if TYPE_CHECKING:
    from magos.registry.state import RegistryState
    from magos.routing.request import RoutedRequest


class NamedValue(_Frozen):
    name: str = Field(min_length=1)
    value: str


class SetModel(_Frozen, Rewriter):
    set_model: str = Field(min_length=1)

    def apply(
        self,
        req: RoutedRequest,
        *,
        registry: RegistryState | None = None,
    ) -> RoutedRequest:
        new_body = dict(req.body)
        new_body["model"] = self.set_model
        return replace(req, body=new_body, body_dirty=True)


class SetHeader(_Frozen, Rewriter):
    set_header: NamedValue

    def apply(
        self,
        req: RoutedRequest,
        *,
        registry: RegistryState | None = None,
    ) -> RoutedRequest:
        return replace(
            req, headers=_with_header(req.headers, self.set_header.name, self.set_header.value)
        )


class RemoveHeader(_Frozen, Rewriter):
    remove_header: str = Field(min_length=1)

    def apply(
        self,
        req: RoutedRequest,
        *,
        registry: RegistryState | None = None,
    ) -> RoutedRequest:
        key = self.remove_header.lower()
        if key not in req.headers:
            return req
        new_headers = dict(req.headers)
        del new_headers[key]
        return replace(req, headers=new_headers)


class AddHeader(_Frozen, Rewriter):
    add_header: NamedValue

    def apply(
        self,
        req: RoutedRequest,
        *,
        registry: RegistryState | None = None,
    ) -> RoutedRequest:
        """No-op when the header is already present (set-if-absent semantics)."""
        key = self.add_header.name.lower()
        if key in req.headers:
            return req
        return replace(
            req, headers=_with_header(req.headers, self.add_header.name, self.add_header.value)
        )


class JqPatch(_Frozen, Rewriter):
    jq_patch: str = Field(min_length=1)

    def apply(
        self,
        req: RoutedRequest,
        *,
        registry: RegistryState | None = None,
    ) -> RoutedRequest:
        from magos.routing.rewrites.jq_patch import RewriteError  # noqa: PLC0415

        result: Any = evaluate_patch(self.jq_patch, dict(req.body))
        if not isinstance(result, Mapping):
            raise RewriteError(
                f"jq_patch result must be a JSON object, got {type(result).__name__}: {self.jq_patch!r}"
            )
        return replace(req, body=dict(result), body_dirty=True)


CompressMode = Literal["token", "cache"]


class CompressOptions(_Frozen):
    """Compression knobs; see ``docs/headroom/pipeline.md``."""

    engine: CompressMode = "token"
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


# Endpoints whose body has a ``messages`` array compatible with Headroom's
# pipeline. /v1/responses uses ``input`` instead and is handled separately.
_COMPRESS_SUPPORTED_ENDPOINTS: frozenset[str] = frozenset(
    {"/v1/messages", "/v1/messages/count_tokens", "/v1/chat/completions"}
)


class Compress(_Frozen, Rewriter):
    compress: CompressOptions = Field(default_factory=CompressOptions)

    def apply(
        self,
        req: RoutedRequest,
        *,
        registry: RegistryState | None = None,
    ) -> RoutedRequest:
        from magos.compression.engine import (  # noqa: PLC0415
            CacheCompressor,
            ResponsesCompressor,
            TokenCompressor,
        )
        from magos.telemetry import get_logger  # noqa: PLC0415

        log = get_logger("magos.routing.rewrites")

        opts = self.compress
        if req.endpoint == "/v1/responses":
            return ResponsesCompressor(opts).apply(req, registry=registry)

        if req.endpoint not in _COMPRESS_SUPPORTED_ENDPOINTS:
            log.debug("compress.skipped_endpoint", endpoint=req.endpoint)
            return req

        messages = req.body.get("messages")
        if not isinstance(messages, list) or not messages:
            return req

        if opts.engine == "cache":
            return CacheCompressor(opts).apply(req, registry=registry)

        return TokenCompressor(opts).apply(req, registry=registry)


def _with_header(headers: Mapping[str, str], name: str, value: str) -> dict[str, str]:
    new_headers = dict(headers)
    new_headers[name.lower()] = value
    return new_headers
