"""Shared helpers + module constants for the e2e suite.

The skip-gate (`MAGOS_E2E=1` + the `e2e` marker) is applied
repo-wide via `tests.e2e.conftest`.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from magos.api import build_api

MODEL = os.environ.get("MAGOS_E2E_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.environ.get("MAGOS_E2E_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
PROMPT = "Reply with the single word: pong"


def anthropic_translate_app() -> Any:
    """Build a magos app that forces `gateway: translate` for /v1/messages.

    The shipped config routes claude-* through byte-exact passthrough,
    which Anthropic's native API rejects when the inbound auth is an OAuth
    access token (`sk-ant-oat*`) without the Claude-Code-only beta
    headers. The translate path goes through `litellm.anthropic_messages`
    instead, which knows to send OAuth as `Authorization: Bearer`. Used
    by tool-use tests that need a working Anthropic upstream regardless of
    the inbound key shape.
    """
    from magos.routing import RoutingConfig  # noqa: PLC0415

    cfg = RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/messages"}},
                    "target": {
                        "provider": "anthropic",
                        "gateway": "translate",
                        "api_key_env": "ANTHROPIC_API_KEY",
                    },
                }
            ]
        }
    )
    return build_api(routing=cfg)


def maybe_skip_anthropic_oauth() -> None:
    """Skip if `ANTHROPIC_API_KEY` is unset; OAuth is fine for translate path."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")


def anthropic_inbound_headers() -> dict[str, str]:
    """Headers a Claude-Code-style client sends on /v1/messages.

    The byte-exact passthrough route forwards inbound headers verbatim;
    the Anthropic upstream rejects OAuth tokens unless `anthropic-beta:
    oauth-2025-04-20` is present alongside `anthropic-version` and the
    bearer. Plain `sk-ant-api03-*` keys go via `x-api-key` and don't
    need the beta. Empirically verified against api.anthropic.com.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    headers: dict[str, str] = {"anthropic-version": "2023-06-01"}
    if api_key.startswith("sk-ant-oat"):
        headers["Authorization"] = f"Bearer {api_key}"
        headers["anthropic-beta"] = "oauth-2025-04-20"
    elif api_key:
        headers["x-api-key"] = api_key
    return headers
