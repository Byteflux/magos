"""``TrackerStore`` get-or-create semantics + TTL eviction."""

from __future__ import annotations

from typing import Any

import pytest

from magos.compression.tracker import PrefixFreezeConfig, TrackerStore, get_store


def test_same_session_id_same_provider_returns_same_instance() -> None:
    store = TrackerStore()
    a = store.get_or_create("derived:abc", "anthropic")
    b = store.get_or_create("derived:abc", "anthropic")
    assert a is b


def test_same_session_id_different_provider_returns_distinct_instances() -> None:
    store = TrackerStore()
    a = store.get_or_create("derived:abc", "anthropic")
    b = store.get_or_create("derived:abc", "openai")
    assert a is not b


def test_different_session_id_returns_distinct_instances() -> None:
    store = TrackerStore()
    a = store.get_or_create("derived:abc", "anthropic")
    b = store.get_or_create("derived:xyz", "anthropic")
    assert a is not b


def test_module_level_store_is_shared() -> None:
    assert get_store() is get_store()


def test_ttl_eviction_drops_idle_trackers(monkeypatch: pytest.MonkeyPatch) -> None:
    store = TrackerStore(config=PrefixFreezeConfig(session_ttl_seconds=10))

    fake_clock = [1000.0]

    def now() -> float:
        return fake_clock[0]

    monkeypatch.setattr("magos.compression.tracker.store.time.time", now)

    a = store.get_or_create("derived:abc", "anthropic")

    # Advance past TTL
    fake_clock[0] += 100.0
    b = store.get_or_create("derived:abc", "anthropic")
    assert b is not a, "expected a fresh tracker after TTL expiry"


def test_get_or_create_passes_freeze_config() -> None:
    cfg = PrefixFreezeConfig(min_cached_tokens=2048, session_ttl_seconds=900)
    store = TrackerStore(config=cfg)
    tracker = store.get_or_create("derived:abc", "anthropic")
    assert tracker.config.min_cached_tokens == 2048
    assert tracker.config.session_ttl_seconds == 900


def test_concurrent_get_or_create_returns_one_instance() -> None:
    """Same key from many threads -> single instance, no duplicate construction."""
    import threading  # noqa: PLC0415

    store = TrackerStore()
    seen: list[Any] = []

    def worker() -> None:
        seen.append(store.get_or_create("derived:concurrent", "anthropic"))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len({id(t) for t in seen}) == 1
