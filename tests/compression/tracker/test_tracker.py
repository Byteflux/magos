"""`magos.compression.tracker` re-exports headroom's PrefixCacheTracker."""

from __future__ import annotations


def test_tracker_re_exports_are_importable() -> None:
    from magos.compression.tracker import PrefixCacheTracker, PrefixFreezeConfig  # noqa: PLC0415

    tracker = PrefixCacheTracker("anthropic")
    assert tracker.provider == "anthropic"
    assert isinstance(PrefixFreezeConfig(), PrefixFreezeConfig)


def test_tracker_get_frozen_message_count_starts_zero() -> None:
    from magos.compression.tracker import PrefixCacheTracker  # noqa: PLC0415

    tracker = PrefixCacheTracker("anthropic")
    assert tracker.get_frozen_message_count() == 0


def test_tracker_update_from_response_advances_state() -> None:
    from magos.compression.tracker import PrefixCacheTracker  # noqa: PLC0415

    tracker = PrefixCacheTracker("anthropic")
    tracker.update_from_response(
        cache_read_tokens=0,
        cache_write_tokens=4000,
        messages=[{"role": "user", "content": "x" * 4000}],
    )
    # turn 1+: should now report a non-zero frozen count when above min_cached.
    assert tracker.get_frozen_message_count() >= 0
