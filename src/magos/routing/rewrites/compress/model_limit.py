"""Context-window limit resolution for the compress rewrite.

Resolves the max input-token window for a dispatch model via the registry,
LiteLLM's bundled registry, or a hardcoded fallback default.
"""

from __future__ import annotations

import contextlib
import io

from magos.registry.state import RegistryState
from magos.telemetry import get_logger

log = get_logger("magos.routing.rewrites")

# Headroom's hardcoded fallback when the caller doesn't supply model_limit
# (`compress.py:161`). Used as our last-resort default when LiteLLM doesn't
# recognise the dispatch model.
_DEFAULT_MODEL_LIMIT = 200_000

# Per-model context-window cache. Populated lazily on first compress call
# for each unique model id. Stores the resolved limit on success AND the
# fallback default on failure, so we don't re-trigger LiteLLM's noisy
# "model not mapped" stderr print on every request for an unknown model.
_MODEL_LIMIT_CACHE: dict[str, int] = {}


def _resolve_model_limit(
    dispatch_model: str,
    *,
    registry: RegistryState | None = None,
    default: int = _DEFAULT_MODEL_LIMIT,
) -> int:
    """Resolve the max input-token window. See `docs/headroom/model-limit.md`."""
    registry_limit = _registry_context_size(dispatch_model, registry)
    if registry_limit is not None:
        return registry_limit

    if dispatch_model in _MODEL_LIMIT_CACHE:
        return _MODEL_LIMIT_CACHE[dispatch_model]

    limit = default
    try:
        # Suppress LiteLLM's noisy stderr provider list on unknown models.
        with contextlib.redirect_stderr(io.StringIO()):
            import litellm  # noqa: PLC0415

            info = litellm.get_model_info(dispatch_model)
    except Exception:
        info = None

    if isinstance(info, dict):
        for key in ("max_input_tokens", "max_tokens"):
            value = info.get(key)
            if isinstance(value, int) and value > 0:
                limit = value
                break

    _MODEL_LIMIT_CACHE[dispatch_model] = limit
    if limit == default:
        log.debug("compress.model_limit_default", dispatch_model=dispatch_model, limit=default)
    else:
        log.debug("compress.model_limit_resolved", dispatch_model=dispatch_model, limit=limit)
    return limit


def _registry_context_size(model: str, registry: RegistryState | None) -> int | None:
    """Registry `context_size` for `model` if known."""
    if registry is None:
        return None
    entry = registry.find_by_model_id(model)
    if entry is None:
        return None
    return entry.context_size
