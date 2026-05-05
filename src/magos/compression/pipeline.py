"""Wrap ``TransformPipeline.apply`` with an inflation guard.

Returns a magos-owned ``ApplyResult``: callers don't depend on
headroom's transform-result type, so additions to ours stay local.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from magos.telemetry import get_logger

from .build import ProviderName
from .config import PipelineConfig
from .registry import get_registry

log = get_logger("magos.compression")


@dataclass(frozen=True, slots=True)
class ApplyResult:
    """Outcome of one ``apply`` call.

    ``inflation_reverted`` is True when the pipeline produced more tokens
    than it received and the wrapper swapped the result back to the
    original messages.
    """

    messages: list[dict[str, Any]]
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    transforms_applied: list[str] = field(default_factory=list)
    inflation_reverted: bool = False


def apply(
    *,
    messages: list[dict[str, Any]],
    model: str,
    model_limit: int,
    config: PipelineConfig,
    provider_name: ProviderName,
    context: str | None = None,
    biases: dict[str, float] | None = None,
) -> ApplyResult:
    """Run the pipeline for ``(config, provider_name)`` against ``messages``.

    On token inflation (``tokens_after > tokens_before``), discards the
    pipeline's output and returns the original messages with zero savings.
    """
    pipeline = get_registry().get_or_build(config, provider_name=provider_name)

    kwargs: dict[str, Any] = {
        "messages": messages,
        "model": model,
        "model_limit": model_limit,
    }
    if context is not None:
        kwargs["context"] = context
    if biases is not None:
        kwargs["biases"] = biases

    raw = pipeline.apply(**kwargs)

    tokens_before = int(getattr(raw, "tokens_before", 0))
    tokens_after = int(getattr(raw, "tokens_after", 0))
    transforms_applied = list(getattr(raw, "transforms_applied", []))
    raw_messages = getattr(raw, "messages", messages)

    if tokens_after > tokens_before > 0:
        log.warning(
            "compress.inflation_reverted",
            model=model,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            transforms=transforms_applied,
        )
        return ApplyResult(
            messages=messages,
            tokens_before=tokens_before,
            tokens_after=tokens_before,
            tokens_saved=0,
            transforms_applied=transforms_applied,
            inflation_reverted=True,
        )

    return ApplyResult(
        messages=raw_messages,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        tokens_saved=max(0, tokens_before - tokens_after),
        transforms_applied=transforms_applied,
        inflation_reverted=False,
    )
