"""Responses-endpoint compression engine step.

Handles cache-aligning the `/v1/responses` `instructions` field.
Token mode is unsupported on this endpoint.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

from magos.compression.engine.base import Compressor
from magos.compression.engine.cache import _run_cache_aligner
from magos.registry.state import RegistryState
from magos.routing.request import RoutedRequest
from magos.routing.schema.rewrites import CompressOptions
from magos.telemetry import get_logger

log = get_logger("magos.routing.rewrites")


class ResponsesCompressor(Compressor):
    """Cache-align the `/v1/responses` `instructions` field; token mode unsupported."""

    def __init__(self, opts: CompressOptions) -> None:
        self._opts = opts

    def apply(
        self,
        req: RoutedRequest,
        *,
        registry: RegistryState | None = None,
    ) -> RoutedRequest:
        opts = self._opts
        if opts.engine != "cache":
            log.debug(
                "compress.responses_token_mode_unsupported",
                endpoint=req.endpoint,
                hint="use engine: cache to stabilise the instructions prefix",
            )
            return req

        instructions = req.body.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            return req

        model = str(req.body.get("model", "")) or "gpt-4o"
        # Wrap as a synthetic system message so the aligner's system-prompt
        # branch fires; we read the mutated content back into `instructions`.
        synthetic = [{"role": "system", "content": instructions}]
        result = _run_cache_aligner(synthetic, model, endpoint=req.endpoint)
        if result is None:
            return req

        new_instructions = result.messages[0].get("content")
        if not isinstance(new_instructions, str) or new_instructions == instructions:
            return req

        log.info(
            "compress.applied",
            endpoint=req.endpoint,
            mode="cache",
            field="instructions",
            transforms=dict(Counter(result.transforms_applied)),
        )
        new_body = dict(req.body)
        new_body["instructions"] = new_instructions
        return replace(req, body=new_body, body_dirty=True)
