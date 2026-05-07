"""``is_ccr_request`` recognises the CCR tool in body.tools across provider shapes."""

from __future__ import annotations

from typing import Any

from magos.compression.ccr import is_ccr_request
from magos.routing.request import RoutedRequest


def _make_req(body: dict[str, Any]) -> RoutedRequest:
    return RoutedRequest(
        endpoint="/v1/messages",
        headers={},
        body=body,
        raw_body=b"",
    )


def test_no_tools_means_not_ccr() -> None:
    req = _make_req({"model": "x"})
    assert is_ccr_request(req) is False


def test_empty_tools_means_not_ccr() -> None:
    req = _make_req({"model": "x", "tools": []})
    assert is_ccr_request(req) is False


def test_anthropic_shape_with_ccr_tool_is_ccr() -> None:
    """Anthropic tools have a top-level ``name`` field."""
    req = _make_req({"model": "x", "tools": [{"name": "headroom_retrieve", "description": "..."}]})
    assert is_ccr_request(req) is True


def test_openai_shape_with_ccr_tool_is_ccr() -> None:
    """OpenAI tools have ``function.name``."""
    req = _make_req(
        {
            "model": "x",
            "tools": [{"type": "function", "function": {"name": "headroom_retrieve"}}],
        }
    )
    assert is_ccr_request(req) is True


def test_anthropic_shape_with_other_tool_is_not_ccr() -> None:
    req = _make_req({"model": "x", "tools": [{"name": "Bash"}]})
    assert is_ccr_request(req) is False


def test_openai_shape_with_other_tool_is_not_ccr() -> None:
    req = _make_req({"model": "x", "tools": [{"type": "function", "function": {"name": "Bash"}}]})
    assert is_ccr_request(req) is False


def test_mixed_tools_with_ccr_present_is_ccr() -> None:
    """CCR is detected even when other tools are also present."""
    req = _make_req(
        {
            "model": "x",
            "tools": [
                {"name": "Bash"},
                {"name": "headroom_retrieve"},
                {"name": "Read"},
            ],
        }
    )
    assert is_ccr_request(req) is True


def test_tools_not_a_list_is_not_ccr() -> None:
    """Defensive: malformed tools field doesn't crash the detector."""
    req = _make_req({"model": "x", "tools": "not-a-list"})
    assert is_ccr_request(req) is False
