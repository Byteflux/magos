"""Compression engine steps: ``Compressor`` ABC and its three concrete classes.

These are constructed per-call inside ``Compress.apply``; the heavy work
(pipeline cache, model_limit cache) already lives in ``magos.compression``.
Phase C3b can hoist construction to config-load time.
"""

from __future__ import annotations

from magos.compression.engine.base import Compressor
from magos.compression.engine.cache import CacheCompressor
from magos.compression.engine.responses import ResponsesCompressor
from magos.compression.engine.token import TokenCompressor

__all__ = [
    "CacheCompressor",
    "Compressor",
    "ResponsesCompressor",
    "TokenCompressor",
]
