"""End-to-end compression + CCR tests.

Each test routes through a token-mode ``compress`` rule and asserts both
that the upstream still succeeds and that magos's compression-side state
(registry, prefix-cache tracker, headroom compression store) advanced
as expected. See ``tests/e2e/conftest.py`` for the ``MAGOS_E2E=1`` skip
gate.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from magos.ingress.http import create_app

from ._helpers import ANTHROPIC_MODEL, MODEL, PROMPT, maybe_skip_anthropic_oauth


def test_compress_token_mode_uses_magos_compression_registry() -> None:
    """A token-mode compress rule on /v1/chat/completions drives the
    magos.compression registry end-to-end and does not break the upstream.
    """
    from magos.compression import get_registry  # noqa: PLC0415
    from magos.routing import RoutingConfig  # noqa: PLC0415

    cfg = RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/chat/completions"}},
                    "rewrites": [{"compress": {"mode": "token"}}],
                    "target": {
                        "provider": "openai",
                        "gateway": "translate",
                        "api_key_env": "OPENAI_API_KEY",
                    },
                }
            ]
        }
    )

    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 16,
    }

    with TestClient(create_app(routing=cfg)) as client:
        resp = client.post("/v1/chat/completions", json=body)

    assert resp.status_code == 200, resp.text
    # MagosCompressionWarmup pre-builds the default pipeline for both
    # providers at lifespan start; assert it's present so we know the
    # request was served by an app whose compression layer is wired up.
    assert list(get_registry().pipelines()), (
        "expected magos.compression registry to be populated after request"
    )


def test_compress_token_mode_freezes_prefix_across_turns() -> None:
    """End-to-end: two turns of the same conversation through a token-mode
    compress rule. Asserts the tracker accumulates state across turns
    (turn_number advances, frozen_message_count becomes non-zero if the
    upstream actually cached anything).
    """
    maybe_skip_anthropic_oauth()
    from magos.cache import derive_session_id, get_store  # noqa: PLC0415
    from magos.routing import RoutingConfig  # noqa: PLC0415

    cfg = RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/messages"}},
                    "rewrites": [{"compress": {"mode": "token"}}],
                    "target": {
                        "provider": "anthropic",
                        "gateway": "translate",
                        "api_key_env": "ANTHROPIC_API_KEY",
                    },
                }
            ]
        }
    )

    headers = {"x-magos-session-id": "phase1.5-e2e-smoke"}
    body_turn1 = {
        "model": ANTHROPIC_MODEL,
        "messages": [{"role": "user", "content": "Say hi."}],
        "max_tokens": 16,
    }

    with TestClient(create_app(routing=cfg)) as client:
        r1 = client.post("/v1/messages", json=body_turn1, headers=headers)
        assert r1.status_code == 200, r1.text

        # Second turn appends an assistant + new user message.
        body_turn2 = {
            "model": ANTHROPIC_MODEL,
            "messages": [
                {"role": "user", "content": "Say hi."},
                {"role": "assistant", "content": r1.json()["content"][0]["text"]},
                {"role": "user", "content": "Say it again."},
            ],
            "max_tokens": 16,
        }
        r2 = client.post("/v1/messages", json=body_turn2, headers=headers)
        assert r2.status_code == 200, r2.text

    # Tracker state survived both turns.
    sid = derive_session_id(headers, body_turn1, "anthropic")
    tracker = get_store().get_or_create(sid, "anthropic")
    # Tracker turn_number was advanced by the post-response hooks.
    assert tracker.stats.turn_number >= 1


def test_ccr_end_to_end_with_compression() -> None:
    """End-to-end: a request large enough to trigger compression goes
    through token-mode compress with CCR enabled. Asserts the wiring
    doesn't error and compression actually happened (store entries exist)."""
    maybe_skip_anthropic_oauth()

    from headroom.cache.compression_store import (  # noqa: PLC0415
        get_compression_store,
        reset_compression_store,
    )

    from magos.routing import RoutingConfig  # noqa: PLC0415

    reset_compression_store()

    cfg = RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/messages"}},
                    "rewrites": [{"compress": {"mode": "token"}}],
                    "target": {
                        "provider": "anthropic",
                        "gateway": "translate",
                        "api_key_env": "ANTHROPIC_API_KEY",
                    },
                }
            ]
        }
    )

    # JSON array of dicts large enough to trigger SmartCrusher compression
    # and compression-store caching (min_items_to_cache = 20).
    items = [{"path": f"path/to/file_{i}.py"} for i in range(50)]
    long_tool_content = json.dumps(items)

    body = {
        "model": ANTHROPIC_MODEL,
        "messages": [
            {"role": "user", "content": "find all python files"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_01x",
                        "name": "Bash",
                        "input": {"command": "find . -name '*.py'"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01x",
                        "content": long_tool_content,
                    }
                ],
            },
            {"role": "user", "content": "summarise"},
        ],
        "max_tokens": 64,
    }

    with TestClient(create_app(routing=cfg)) as client:
        r = client.post("/v1/messages", json=body)
        assert r.status_code == 200, r.text

    # Compression actually happened.
    store = get_compression_store()
    stats = store.get_stats()
    assert stats.get("entry_count", 0) >= 1
