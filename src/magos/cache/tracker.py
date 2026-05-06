"""Re-export ``headroom.cache.prefix_tracker`` types under the magos namespace.

Callers import ``PrefixCacheTracker`` and ``PrefixFreezeConfig`` from here
rather than directly from headroom; gives us a single seam if the upstream
API ever changes.
"""

from __future__ import annotations

from headroom.cache.prefix_tracker import PrefixCacheTracker, PrefixFreezeConfig

__all__ = ["PrefixCacheTracker", "PrefixFreezeConfig"]
