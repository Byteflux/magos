"""Field-precedence merge across override / discovery / litellm sources.

Precedence (highest first):

    1. override   - operator-authored values in ``magos.yaml``
    2. discovery  - live values from the provider's discovery adapter
    3. litellm    - bundled-registry fallback via ``litellm.get_model_info``

For each field we walk the chain in order and take the first non-``None``
value. ``sources`` on the final ``ModelEntry`` records the contributors
that supplied at least one field, in priority order, as audit trail.

The merge is pure: it only consumes ``PartialEntry`` values plus the
identifiers (``provider``, ``raw_id``, ``litellm_id`` adapter default)
the caller already resolved upstream.
"""

from __future__ import annotations

from collections.abc import Iterable

from magos.registry.litellm_lookup import PartialEntry
from magos.registry.models import ModelEntry

_SOURCE_ORDER: tuple[str, ...] = ("override", "discovery", "litellm")


def _pick_first(parts: Iterable[PartialEntry], attr: str) -> object | None:
    for part in parts:
        value: object | None = getattr(part, attr)
        if value is not None:
            return value
    return None


def merge(
    *,
    provider: str,
    raw_id: str,
    default_litellm_id: str,
    override: PartialEntry | None = None,
    discovered: PartialEntry | None = None,
    litellm_fallback: PartialEntry | None = None,
) -> ModelEntry:
    """Combine partials into a ``ModelEntry`` honoring precedence.

    ``default_litellm_id`` is the adapter-computed dispatch id used when
    no source supplies a ``litellm_id``. The override layer can replace it.
    """
    chain_named: tuple[tuple[str, PartialEntry | None], ...] = (
        ("override", override),
        ("discovery", discovered),
        ("litellm", litellm_fallback),
    )
    parts: tuple[PartialEntry, ...] = tuple(p for _, p in chain_named if p is not None)

    litellm_id = _pick_first(parts, "litellm_id")
    if litellm_id is None:
        litellm_id = default_litellm_id

    sources: list[str] = []
    for name, part in chain_named:
        if part is None:
            continue
        if any(getattr(part, attr) is not None for attr in _ENTRY_FIELDS):
            sources.append(name)

    context_size = _pick_first(parts, "context_size")
    max_output = _pick_first(parts, "max_output")
    input_cost = _pick_first(parts, "input_cost")
    output_cost = _pick_first(parts, "output_cost")
    modalities = _pick_first(parts, "modalities")

    return ModelEntry(
        provider=provider,
        raw_id=raw_id,
        litellm_id=str(litellm_id),
        context_size=context_size if isinstance(context_size, int) else None,
        max_output=max_output if isinstance(max_output, int) else None,
        input_cost=float(input_cost) if isinstance(input_cost, (int, float)) else None,
        output_cost=float(output_cost) if isinstance(output_cost, (int, float)) else None,
        modalities=tuple(modalities) if isinstance(modalities, tuple) else (),
        sources=tuple(sources),
    )


_ENTRY_FIELDS: tuple[str, ...] = (
    "litellm_id",
    "context_size",
    "max_output",
    "input_cost",
    "output_cost",
    "modalities",
)


__all__ = ["_SOURCE_ORDER", "merge"]
