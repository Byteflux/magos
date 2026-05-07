"""mitmproxy addon: passive structured logging of outbound LLM traffic.

Loaded in-process alongside `MagosIngressAddon`; can also run out-of-
process via `mitmdump -s ... --listen-port 8080` with magos pointed at
it via `HTTPS_PROXY`. Modifies nothing; logs flows whose host matches a
known LLM provider. See `docs/ingress.md`.
"""

from __future__ import annotations

import time

from mitmproxy import http

from magos.telemetry import get_logger

log = get_logger("magos.proxy.addons.observer")

LLM_PROVIDER_HOSTS: frozenset[str] = frozenset(
    {
        "api.openai.com",
        "api.anthropic.com",
        "generativelanguage.googleapis.com",
        "api.cohere.ai",
        "api.cohere.com",
        "api.mistral.ai",
        "api.groq.com",
        "api.together.xyz",
        "api.deepseek.com",
        "api.fireworks.ai",
        "api.perplexity.ai",
    }
)

_START_KEY = "magos_request_started_at"


def _is_llm_host(host: str) -> bool:
    """Return True for known LLM provider hosts and their subdomains."""
    if host in LLM_PROVIDER_HOSTS:
        return True
    return any(host.endswith(f".{h}") for h in LLM_PROVIDER_HOSTS)


class MagosObserverAddon:
    """Logs request/response metadata for outbound LLM provider traffic."""

    def request(self, flow: http.HTTPFlow) -> None:
        host = flow.request.pretty_host
        if not _is_llm_host(host):
            return
        flow.metadata[_START_KEY] = time.monotonic()
        log.info(
            "egress.request",
            host=host,
            method=flow.request.method,
            path=flow.request.path,
            scheme=flow.request.scheme,
            content_length=len(flow.request.raw_content or b""),
        )

    def response(self, flow: http.HTTPFlow) -> None:
        host = flow.request.pretty_host
        if not _is_llm_host(host) or flow.response is None:
            return
        start = flow.metadata.get(_START_KEY)
        latency_ms: float | None = None
        if isinstance(start, float):
            latency_ms = round((time.monotonic() - start) * 1000.0, 2)
        log.info(
            "egress.response",
            host=host,
            method=flow.request.method,
            path=flow.request.path,
            status=flow.response.status_code,
            latency_ms=latency_ms,
            content_length=len(flow.response.raw_content or b""),
        )


addons = [MagosObserverAddon()]
