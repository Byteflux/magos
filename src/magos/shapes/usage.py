"""`Usage`: canonicalised token counts for one request, shape-agnostic."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Usage:
    """Canonicalised token counts for one request.

    `cache_write` is Anthropic-only; OpenAI shapes leave it 0.
    """

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0

    @property
    def is_empty(self) -> bool:
        return (
            self.input == 0 and self.output == 0 and self.cache_read == 0 and self.cache_write == 0
        )
