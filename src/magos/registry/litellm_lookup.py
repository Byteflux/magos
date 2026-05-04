"""LiteLLM bundled-registry fallback lookup.

LiteLLM ships a JSON registry of known models with context windows, costs,
and modality/capability flags. We use it as the lowest-precedence source
in the merge chain: when a provider's discovery omits desired fields and
no operator override is set, we fall back to whatever LiteLLM knows.

``PartialEntry`` is the normalized shape every source produces; merge then
layers them by precedence into a final ``ModelEntry``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import litellm

from magos.telemetry import get_logger

log = get_logger("magos.registry.litellm_lookup")


@dataclass(frozen=True, slots=True)
class PartialEntry:
    """Source-agnostic partial fields for a model.

    Every discovery adapter, the litellm fallback, and the override layer
    all produce ``PartialEntry`` values; the merge function combines them
    by precedence into a fully-formed ``ModelEntry``.
    """

    litellm_id: str | None = None
    context_size: int | None = None
    max_output: int | None = None
    # USD per million tokens. Adapters scale upstream per-token values.
    input_cost: float | None = None
    output_cost: float | None = None
    cache_read_cost: float | None = None
    cache_write_cost: float | None = None
    modalities: tuple[str, ...] | None = None


class GetModelInfoFn(Protocol):
    """Injection seam: production wires ``litellm.get_model_info``."""

    def __call__(self, model: str) -> dict[str, Any]: ...


def _coerce_modalities(info: dict[str, Any]) -> tuple[str, ...] | None:
    """Derive a modality tuple from LiteLLM's capability flags.

    LiteLLM doesn't expose modalities as a list; it exposes booleans like
    ``supports_vision``, ``supports_audio_input``. We translate to a small
    fixed vocabulary so registry consumers can rely on string membership.
    """
    modalities: list[str] = ["text"]
    if info.get("supports_vision"):
        modalities.append("image")
    if info.get("supports_audio_input"):
        modalities.append("audio")
    return tuple(modalities) if modalities else None


def lookup(litellm_id: str, *, get_info: GetModelInfoFn | None = None) -> PartialEntry:
    """Best-effort lookup of ``litellm_id`` against LiteLLM's bundled registry.

    Returns an empty ``PartialEntry`` if LiteLLM doesn't know the model
    (ValueError) or any other lookup error; we log at debug because misses
    are expected for non-mainline providers and shouldn't be alarming.
    """
    fn = get_info or litellm.get_model_info
    try:
        info = fn(model=litellm_id)
    except (ValueError, KeyError) as exc:
        log.debug("registry.litellm_lookup.miss", model=litellm_id, error=str(exc))
        return PartialEntry()
    except Exception as exc:
        # LiteLLM raises a bare Exception with "isn't mapped yet" for unknown
        # models, treat that as an expected miss; anything else is a real error.
        msg = str(exc)
        if "isn't mapped yet" in msg or "not mapped yet" in msg:
            log.debug("registry.litellm_lookup.miss", model=litellm_id, error=msg)
        else:
            log.warning(
                "registry.litellm_lookup.error",
                model=litellm_id,
                error=msg,
                error_type=type(exc).__name__,
            )
        return PartialEntry()
    info_dict: dict[str, Any] = dict(info)
    return PartialEntry(
        litellm_id=litellm_id,
        context_size=info_dict.get("max_input_tokens") or info_dict.get("max_tokens"),
        max_output=info_dict.get("max_output_tokens"),
        # LiteLLM reports USD per token; magos tracks USD per million tokens.
        input_cost=_per_token_to_per_million(info_dict.get("input_cost_per_token")),
        output_cost=_per_token_to_per_million(info_dict.get("output_cost_per_token")),
        cache_read_cost=_per_token_to_per_million(info_dict.get("cache_read_input_token_cost")),
        cache_write_cost=_per_token_to_per_million(
            info_dict.get("cache_creation_input_token_cost")
        ),
        modalities=_coerce_modalities(info_dict),
    )


def _per_token_to_per_million(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    return float(value) * 1_000_000
