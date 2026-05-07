"""`derive_session_id` truth table: explicit header wins; derivation is stable."""

from __future__ import annotations

from typing import Any

from magos.compression.tracker import derive_session_id


def _body(model: str = "claude-sonnet-4-5", system: Any = "You are helpful.") -> dict[str, Any]:
    return {"model": model, "system": system, "messages": [{"role": "user", "content": "hi"}]}


def test_explicit_header_wins() -> None:
    sid = derive_session_id({"x-magos-session-id": "client-conv-42"}, _body(), "anthropic")
    assert sid == "explicit:client-conv-42"


def test_explicit_header_empty_falls_through_to_derived() -> None:
    sid = derive_session_id({"x-magos-session-id": "  "}, _body(), "anthropic")
    assert sid.startswith("derived:")


def test_explicit_header_strips_whitespace() -> None:
    sid = derive_session_id({"x-magos-session-id": "  conv-42  "}, _body(), "anthropic")
    assert sid == "explicit:conv-42"


def test_derivation_is_deterministic() -> None:
    a = derive_session_id({}, _body(), "anthropic")
    b = derive_session_id({}, _body(), "anthropic")
    assert a == b
    assert a.startswith("derived:")


def test_derivation_provider_distinct() -> None:
    a = derive_session_id({}, _body(), "anthropic")
    b = derive_session_id({}, _body(), "openai")
    assert a != b


def test_derivation_model_distinct() -> None:
    a = derive_session_id({}, _body(model="claude-sonnet-4-5"), "anthropic")
    b = derive_session_id({}, _body(model="claude-opus-4"), "anthropic")
    assert a != b


def test_derivation_system_distinct() -> None:
    a = derive_session_id({}, _body(system="You are helpful."), "anthropic")
    b = derive_session_id({}, _body(system="You are concise."), "anthropic")
    assert a != b


def test_derivation_uses_authorization_bearer_prefix() -> None:
    headers_a = {"authorization": "Bearer sk-ant-api03-AAAAAAAA-rest-of-key"}
    headers_b = {"authorization": "Bearer sk-ant-api03-BBBBBBBB-rest-of-key"}
    a = derive_session_id(headers_a, _body(), "anthropic")
    b = derive_session_id(headers_b, _body(), "anthropic")
    assert a != b


def test_derivation_uses_x_api_key_when_no_bearer() -> None:
    a = derive_session_id({"x-api-key": "sk-ant-api03-XXXXXXXX-tail"}, _body(), "anthropic")
    b = derive_session_id({"x-api-key": "sk-ant-api03-YYYYYYYY-tail"}, _body(), "anthropic")
    assert a != b


def test_derivation_with_no_auth_header_collapses_unauthed_clients() -> None:
    a = derive_session_id({}, _body(), "anthropic")
    b = derive_session_id({}, _body(), "anthropic")
    assert a == b


def test_derivation_with_missing_model_uses_unknown_marker() -> None:
    sid = derive_session_id({}, {"system": "x", "messages": []}, "anthropic")
    assert sid.startswith("derived:")


def test_derivation_with_missing_system_uses_empty_bytes() -> None:
    sid_no_system = derive_session_id({}, {"model": "x", "messages": []}, "anthropic")
    sid_empty_system = derive_session_id(
        {}, {"model": "x", "system": "", "messages": []}, "anthropic"
    )
    assert sid_no_system == sid_empty_system


def test_openai_uses_first_system_role_message() -> None:
    body_a = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are A."},
            {"role": "user", "content": "hi"},
        ],
    }
    body_b = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are B."},
            {"role": "user", "content": "hi"},
        ],
    }
    a = derive_session_id({}, body_a, "openai")
    b = derive_session_id({}, body_b, "openai")
    assert a != b
