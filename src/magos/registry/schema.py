"""Pydantic schemas for the registry-related YAML grammar.

Mirrors the new top-level blocks in ``magos.yaml``:

- ``providers:``        per-provider declaration (creds, base_url, discovery
                        adapter override, refresh interval override, optional
                        per-model overrides + manual registration).
- ``provider_order:``   tie-break order when multiple providers serve the
                        same logical model and no explicit pin is set.
- ``registry:``         registry-wide knobs: default refresh interval,
                        on_unknown_model behavior, models.json path.

Variants are frozen + ``extra="forbid"`` so typos surface at config load
time rather than silently ignoring fields.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

OnUnknownModel = Literal["error", "passthrough"]
DiscoveryAdapter = Literal[
    "openai",
    "anthropic",
    "openrouter",
    "vultr",
    "noop",
]


_DURATION_RE = re.compile(r"^\s*(\d+)\s*(ms|s|m|h|d)\s*$")
_UNIT_SECONDS: dict[str, int] = {
    "ms": 0,  # rounded down to int seconds; ms granularity not supported
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
}


def _parse_duration(value: object) -> int:
    """Coerce ``"30s"`` / ``"2h"`` style duration strings into seconds.

    Accepts bare ``int`` for seconds. Negative or zero values are rejected
    by the field constraint, not here, so callers can distinguish "missing"
    from "invalid format" cleanly.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if not isinstance(value, str):
        raise TypeError(
            f"duration must be int seconds or string like '2h', got {type(value).__name__}"
        )
    match = _DURATION_RE.match(value)
    if match is None:
        raise ValueError(f"invalid duration: {value!r}; expected '<int><ms|s|m|h|d>'")
    magnitude = int(match.group(1))
    unit = match.group(2)
    if unit == "ms":
        # Round to nearest second, minimum 1 if the user explicitly asked for ms.
        return max(1, round(magnitude / 1000))
    return magnitude * _UNIT_SECONDS[unit]


DurationSeconds = Annotated[int, BeforeValidator(_parse_duration), Field(ge=1)]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class ModelOverride(_Frozen):
    """Per-model override layered on top of discovery + litellm fallback.

    All fields optional. Unset fields fall through to lower-precedence
    sources during merge. ``litellm_id`` overrides the adapter-default
    dispatch id when a provider's namespacing differs from the adapter's
    expectation.
    """

    context_size: int | None = Field(default=None, ge=1)
    max_output: int | None = Field(default=None, ge=1)
    # USD per million tokens (e.g. 3.0 = $3/M).
    input_cost: float | None = Field(default=None, ge=0)
    output_cost: float | None = Field(default=None, ge=0)
    modalities: tuple[str, ...] | None = None
    litellm_id: str | None = Field(default=None, min_length=1)


class ProviderConfig(_Frozen):
    """One provider entry under the top-level ``providers:`` block.

    ``discovery`` unset or set to ``noop`` means manual-only: no network
    calls, manual entries from ``models`` are permanent until removed
    from yaml. ``litellm_provider`` overrides the adapter default for
    constructing dispatch ids.
    """

    api_key_env: str | None = Field(default=None, min_length=1)
    base_url: str | None = Field(default=None, min_length=1)
    discovery: DiscoveryAdapter | None = None
    refresh_interval: DurationSeconds | None = None
    litellm_provider: str | None = Field(default=None, min_length=1)
    models: dict[str, ModelOverride] = Field(default_factory=dict)


class RegistrySettings(_Frozen):
    """Registry-wide knobs."""

    refresh_interval: DurationSeconds = Field(default=2 * 3600)
    on_unknown_model: OnUnknownModel = "error"
    models_path: str = Field(default="./models.json", min_length=1)
    deprecation_grace_seconds: DurationSeconds = Field(default=3 * 86400)
    discovery_timeout_seconds: DurationSeconds = Field(default=30)
    discovery_max_attempts: int = Field(default=3, ge=1)
    boot_discovery_timeout_seconds: DurationSeconds = Field(default=10)
    boot_discovery_max_attempts: int = Field(default=1, ge=1)


class RegistryYaml(_Frozen):
    """Top-level registry-related blocks parsed from ``magos.yaml``.

    The loader composes this with ``RoutingConfig`` into ``MagosConfig``.
    All fields default to empty so a yaml without any registry blocks
    parses cleanly and the server runs in routing-only mode.
    """

    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    provider_order: tuple[str, ...] = ()
    registry: RegistrySettings = Field(default_factory=RegistrySettings)
