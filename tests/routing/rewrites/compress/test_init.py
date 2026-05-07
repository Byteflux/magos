"""Dispatcher-level ``compress`` rewrite tests: skip/no-op + CCR injection.

These exercise the dispatch logic in ``Compress.apply``: which endpoint shapes
get dispatched where, when the rewrite returns the request untouched, and when
CCR tool injection fires.
"""

from __future__ import annotations

from typing import Any

import pytest

from magos.compression import ApplyResult
from magos.compression.engine import token as tm
from magos.routing import Compress, CompressOptions
from magos.routing.rewrites import apply_transforms
from tests.routing._helpers import make_req

# --- Skip / no-op cases ---


def test_compress_skipped_on_responses_endpoint() -> None:
    req = make_req(
        endpoint="/v1/responses",
        body={"model": "x", "input": "hello"},
    )
    out = apply_transforms(req, [Compress(compress=CompressOptions())])
    assert out is req


def test_compress_no_messages_is_noop() -> None:
    req = make_req(body={"model": "x"})
    out = apply_transforms(req, [Compress(compress=CompressOptions())])
    assert out is req


def test_compress_empty_messages_is_noop() -> None:
    req = make_req(body={"model": "x", "messages": []})
    out = apply_transforms(req, [Compress(compress=CompressOptions())])
    assert out is req


def test_compress_unsupported_endpoint_does_not_call_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:  # pragma: no cover
        raise AssertionError("apply() must not be called for /v1/responses")

    monkeypatch.setattr(tm, "apply", boom, raising=True)

    req = make_req(
        endpoint="/v1/responses",
        body={"model": "x", "input": "hello"},
    )
    out = apply_transforms(req, [Compress(compress=CompressOptions())])
    assert out is req


def test_responses_aux_endpoints_skip_compress() -> None:
    """The /v1/responses/{id} family has no body to compress; must no-op."""
    for endpoint in ("/v1/responses/{id}", "/v1/responses/{id}/input_items"):
        req = make_req(endpoint=endpoint, body={}, raw=b"")
        out = apply_transforms(req, [Compress(compress=CompressOptions(engine="cache"))])
        assert out is req, f"{endpoint} should no-op"


# --- CCR tool injection ---


def test_compress_ccr_injects_tool_when_markers_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When apply() output has compression markers, CCR tool gets injected."""
    from magos.compression.tracker import store as store_mod  # noqa: PLC0415

    def fake_apply(**kwargs: Any) -> ApplyResult:
        return ApplyResult(
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": (
                        "Tool result: [100 items compressed to 10. "
                        "Retrieve more: hash=abcdef0123456789abcdef01]"
                    ),
                },
            ],
            tokens_before=200,
            tokens_after=100,
            tokens_saved=100,
        )

    monkeypatch.setattr(tm, "apply", fake_apply, raising=True)
    monkeypatch.setattr(store_mod, "_STORE", store_mod.TrackerStore())

    req = make_req(
        body={
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "verbose"}],
        }
    )
    out = apply_transforms(req, [Compress(compress=CompressOptions())])

    tools = out.body.get("tools", [])
    tool_names = [t.get("name") for t in tools]
    assert "headroom_retrieve" in tool_names


def test_compress_ccr_disabled_skips_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from magos.compression.tracker import store as store_mod  # noqa: PLC0415

    def fake_apply(**kwargs: Any) -> ApplyResult:
        return ApplyResult(
            messages=[
                {
                    "role": "assistant",
                    "content": (
                        "[100 items compressed to 10. Retrieve more: hash=abcdef0123456789abcdef01]"
                    ),
                }
            ],
            tokens_before=100,
            tokens_after=50,
            tokens_saved=50,
        )

    monkeypatch.setattr(tm, "apply", fake_apply, raising=True)
    monkeypatch.setattr(store_mod, "_STORE", store_mod.TrackerStore())

    req = make_req(
        body={
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "x"}],
        }
    )
    out = apply_transforms(req, [Compress(compress=CompressOptions(ccr_enabled=False))])

    tools = out.body.get("tools", [])
    tool_names = [t.get("name") for t in tools]
    assert "headroom_retrieve" not in tool_names


def test_compress_ccr_no_markers_no_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If apply() output has no compression markers, nothing is injected."""
    from magos.compression.tracker import store as store_mod  # noqa: PLC0415

    def fake_apply(**kwargs: Any) -> ApplyResult:
        return ApplyResult(
            messages=[{"role": "user", "content": "shorter"}],
            tokens_before=100,
            tokens_after=60,
            tokens_saved=40,
        )

    monkeypatch.setattr(tm, "apply", fake_apply, raising=True)
    monkeypatch.setattr(store_mod, "_STORE", store_mod.TrackerStore())

    req = make_req(
        body={
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "verbose"}],
        }
    )
    out = apply_transforms(req, [Compress(compress=CompressOptions())])

    assert "tools" not in out.body or out.body.get("tools") == []
