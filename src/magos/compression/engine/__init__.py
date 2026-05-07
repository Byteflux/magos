"""Compression engine steps: ``Compressor`` ABC and its three concrete classes.

These are constructed per-call inside ``Compress.apply``; the heavy work
(pipeline cache, model_limit cache) already lives in ``magos.compression``.
Phase C3b can hoist construction to config-load time.
"""

from __future__ import annotations

from .base import Compressor
from .cache import CacheCompressor
from .responses import ResponsesCompressor
from .token import TokenCompressor

__all__ = [
    "CacheCompressor",
    "Compressor",
    "ResponsesCompressor",
    "TokenCompressor",
]
