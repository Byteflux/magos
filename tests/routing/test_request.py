"""``RoutedRequest.post_response_hooks`` defaults + reference semantics."""

from __future__ import annotations

from dataclasses import replace

from magos.egress.usage import Usage
from magos.routing.request import RoutedRequest


def _make_req() -> RoutedRequest:
    return RoutedRequest(
        endpoint="/v1/messages",
        headers={},
        body={"model": "x", "messages": []},
        raw_body=b"",
    )


def test_post_response_hooks_defaults_to_empty_list() -> None:
    req = _make_req()
    assert req.post_response_hooks == []


def test_post_response_hooks_default_is_per_instance_not_shared() -> None:
    a = _make_req()
    b = _make_req()
    assert a.post_response_hooks is not b.post_response_hooks


def test_post_response_hooks_list_is_mutable() -> None:
    req = _make_req()

    def hook(_: Usage) -> None:
        return None

    req.post_response_hooks.append(hook)
    assert req.post_response_hooks == [hook]


def test_replace_preserves_hook_list_reference() -> None:
    req = _make_req()

    def hook(_: Usage) -> None:
        return None

    req.post_response_hooks.append(hook)
    new_req = replace(req, body_dirty=True)
    assert new_req.post_response_hooks is req.post_response_hooks
    assert new_req.post_response_hooks == [hook]
