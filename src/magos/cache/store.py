"""Per-(session_id, provider) registry of ``PrefixCacheTracker`` instances.

TTL-based lazy eviction: on each ``get_or_create`` we drop any tracker
whose ``is_expired`` flag is set. Headroom's tracker tracks last-activity
internally so actively-used trackers stay alive past their nominal TTL.
No background cleanup; bounded session count keeps the in-memory dict small.

The ``import time`` is retained so tests can monkeypatch
``magos.cache.store.time.time`` to advance the clock; headroom's
``is_expired`` reads the same module's ``time.time``.
"""

from __future__ import annotations

import threading
import time  # noqa: F401  -- retained for monkeypatching; see module docstring

from headroom.cache.prefix_tracker import PrefixCacheTracker, PrefixFreezeConfig

from magos.compression import ProviderName


class TrackerStore:
    """Session-keyed cache of ``PrefixCacheTracker`` instances.

    Keyed by ``(session_id, provider)``. Expired entries (those whose
    ``is_expired`` flag is set by headroom's last-activity check) are
    evicted lazily on each ``get_or_create`` call; no background thread
    is required.

    Thread-safe via a single ``threading.Lock`` with double-checked locking.
    """

    def __init__(self, config: PrefixFreezeConfig | None = None) -> None:
        self._config = config or PrefixFreezeConfig()
        self._cache: dict[tuple[str, str], PrefixCacheTracker] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str, provider: ProviderName) -> PrefixCacheTracker:
        """Return the cached tracker for ``(session_id, provider)``, or create one.

        If the existing entry has exceeded ``session_ttl_seconds`` of
        idleness, it is discarded and a fresh tracker is constructed.
        """
        key = (session_id, provider)
        cached = self._cache.get(key)
        if cached is not None and not cached.is_expired:
            return cached
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None and not cached.is_expired:
                return cached
            self._evict_expired()
            tracker = PrefixCacheTracker(provider=provider, config=self._config)
            self._cache[key] = tracker
            return tracker

    def _evict_expired(self) -> None:
        """Drop expired trackers; must be called under ``self._lock``."""
        expired = [key for key, t in self._cache.items() if t.is_expired]
        for key in expired:
            del self._cache[key]


_STORE = TrackerStore()


def get_store() -> TrackerStore:
    """Return the process-wide ``TrackerStore`` singleton."""
    return _STORE
