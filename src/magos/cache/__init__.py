"""Magos-owned prefix-cache tracking layer.

Wraps ``headroom.cache.prefix_tracker.PrefixCacheTracker`` with a
session-id-keyed store and TTL eviction. The compress routing rewrite
fetches a tracker per request to read ``frozen_message_count``; the
egress layer fires post-response hooks that update the tracker with
the upstream's reported cache_read / cache_write tokens.
"""

from __future__ import annotations

from headroom.cache.prefix_tracker import PrefixCacheTracker, PrefixFreezeConfig

from .session_id import derive_session_id
from .store import TrackerStore, get_store

__all__ = [
    "PrefixCacheTracker",
    "PrefixFreezeConfig",
    "TrackerStore",
    "derive_session_id",
    "get_store",
]
