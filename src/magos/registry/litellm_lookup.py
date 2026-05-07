"""LiteLLM bundled-registry fallback (lowest-precedence source in merge).

``PartialEntry`` is the normalised shape produced by every source; merge
layers them by precedence into a ``ModelEntry``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import litellm

from magos.registry.discovery._coerce import coerce_float, per_token_to_per_million
from magos.telemetry import get_logger

log = get_logger("magos.registry.litellm_lookup")


@dataclass(frozen=True, slots=True)
class PartialEntry:
    """Source-agnostic partial fields for a model; merge layers them by precedence."""

    litellm_id: str | None = None
    context_size: int | None = None
    max_output: int | None = None
    # USD per million tokens. Adapters scale upstream per-token values.
    input_cost: float | None = None
    output_cost: float | None = None
    cache_read_cost: float | None = None
    cache_write_cost: float | None = None
    input_modalities: tuple[str, ...] | None = None
    output_modalities: tuple[str, ...] | None = None


class GetModelInfoFn(Protocol):
    """Injection seam: production wires ``litellm.get_model_info``."""

    def __call__(self, model: str) -> dict[str, Any]: ...


def _coerce_input_modalities(info: dict[str, Any]) -> tuple[str, ...] | None:
    """Translate LiteLLM ``supports_*`` booleans to a fixed modality tuple."""
    modalities: list[str] = ["text"]
    if info.get("supports_vision"):
        modalities.append("image")
    if info.get("supports_audio_input"):
        modalities.append("audio")
    return tuple(modalities)


def _coerce_output_modalities(info: dict[str, Any]) -> tuple[str, ...] | None:
    modalities: list[str] = ["text"]
    if info.get("supports_audio_output"):
        modalities.append("audio")
    if info.get("supports_image_output"):
        modalities.append("image")
    return tuple(modalities)


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
        input_cost=_per_token(info_dict.get("input_cost_per_token")),
        output_cost=_per_token(info_dict.get("output_cost_per_token")),
        cache_read_cost=_per_token(info_dict.get("cache_read_input_token_cost")),
        cache_write_cost=_per_token(info_dict.get("cache_creation_input_token_cost")),
        input_modalities=_coerce_input_modalities(info_dict),
        output_modalities=_coerce_output_modalities(info_dict),
    )


def _per_token(value: Any) -> float | None:
    """``Any`` -> per-million USD; drops bools, non-numeric, and negatives."""
    return per_token_to_per_million(coerce_float(value))
