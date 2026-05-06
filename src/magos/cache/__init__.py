"""Magos-owned prefix-cache tracking layer.

Wraps ``headroom.cache.prefix_tracker.PrefixCacheTracker`` with a
session-id-keyed store and TTL eviction. The compress routing rewrite
fetches a tracker per request to read ``frozen_message_count``; the
egress layer fires post-response hooks that update the tracker with
the upstream's reported cache_read / cache_write tokens.

See ``docs/superpowers/specs/2026-05-06-phase-1.5-prefix-cache-tracking-design.md``.
"""

from __future__ import annotations

from .session_id import derive_session_id
from .store import TrackerStore, get_store
from .tracker import PrefixCacheTracker, PrefixFreezeConfig

__all__ = [
    "PrefixCacheTracker",
    "PrefixFreezeConfig",
    "TrackerStore",
    "derive_session_id",
    "get_store",
]
