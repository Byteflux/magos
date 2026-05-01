"""Unit tests for the mitmproxy egress observer addon.

The behavioural surface is small: which hosts are matched and which logs are
emitted. We test the host predicate directly and assert log shape via
``structlog.testing.capture_logs``.
"""

from __future__ import annotations

import structlog

from magos.addon import LLM_PROVIDER_HOSTS, MagosObserverAddon, _is_llm_host


def test_is_llm_host_matches_exact() -> None:
    for host in LLM_PROVIDER_HOSTS:
        assert _is_llm_host(host)


def test_is_llm_host_matches_subdomain() -> None:
    assert _is_llm_host("eu.api.openai.com")
    assert _is_llm_host("staging.api.anthropic.com")


def test_is_llm_host_rejects_unknown() -> None:
    assert not _is_llm_host("example.com")
    assert not _is_llm_host("openai.com")  # bare domain not in the list


def test_addon_logs_request_and_response_for_llm_host() -> None:
    addon = MagosObserverAddon()

    class _Req:
        pretty_host = "api.openai.com"
        method = "POST"
        path = "/v1/chat/completions"
        scheme = "https"
        raw_content = b"{}"

    class _Resp:
        status_code = 200
        raw_content = b"{}"

    class _Flow:
        request = _Req()
        response: _Resp | None = None
        metadata: dict[str, object] = {}  # noqa: RUF012

    flow = _Flow()

    with structlog.testing.capture_logs() as logs:
        addon.request(flow)  # type: ignore[arg-type]
        flow.response = _Resp()
        addon.response(flow)  # type: ignore[arg-type]

    events = [e["event"] for e in logs]
    assert events == ["egress.request", "egress.response"]
    assert logs[1]["status"] == 200
    assert isinstance(logs[1]["latency_ms"], float)


def test_addon_ignores_non_llm_host() -> None:
    addon = MagosObserverAddon()

    class _Req:
        pretty_host = "example.com"
        method = "GET"
        path = "/"
        scheme = "https"
        raw_content = b""

    class _Resp:
        status_code = 200
        raw_content = b""

    class _Flow:
        request = _Req()
        response = _Resp()
        metadata: dict[str, object] = {}  # noqa: RUF012

    flow = _Flow()

    with structlog.testing.capture_logs() as logs:
        addon.request(flow)  # type: ignore[arg-type]
        addon.response(flow)  # type: ignore[arg-type]

    assert logs == []
