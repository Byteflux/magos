"""Probe Anthropic prompt caching through magos.

Sends the same long-system-prompt request twice and reports
``cache_creation_input_tokens`` and ``cache_read_input_tokens`` so you
can see whether magos's request rewrites (notably the global ``compress``
pre_rewrite over the Anthropic passthrough rule) preserve cache hits.

Two passes:

1. Through magos (default ``http://127.0.0.1:6246``).
2. Direct to ``api.anthropic.com`` as a control.

For each pass: call once to write the cache, call again to read it. A
working cache shows ``cache_creation`` > 0 on the first call and
``cache_read`` > 0 on the second. If magos's compress rewrite is
deterministic, the magos pass should look identical to the direct pass
(possibly with different absolute token counts because the system
prompt has been compressed, but the same hit/miss pattern).

Usage::

    # start magos in another terminal first:
    uv run magos
    # then:
    ANTHROPIC_API_KEY=sk-ant-... uv run python scripts/probe_anthropic_cache.py

Optional env::

    MAGOS_URL=http://127.0.0.1:6246        (default)
    MODEL=claude-sonnet-4-6                (default; must satisfy Anthropic's
                                            min cacheable tokens for the model)
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Match the e2e/conftest convention: pull .env from the project root
# into the process environment so ANTHROPIC_API_KEY is picked up the same
# way our tests pick it up.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

MAGOS_URL = os.environ.get("MAGOS_URL", "http://127.0.0.1:6246").rstrip("/")
ANTHROPIC_URL = "https://api.anthropic.com"
MODEL = os.environ.get("MODEL", "claude-haiku-4-5-20251001")
API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Anthropic caches prefixes >=1024 tokens for Sonnet/Opus. ~600 repetitions
# of a ~30-char sentence comfortably exceeds that with room for compression
# to remove some content and still leave a cacheable prefix.
SYSTEM_PROMPT = (
    "You are a helpful coding assistant who explains things clearly and concisely. "
    * 1000
)


def _build_body(call_idx: int) -> dict[str, object]:
    return {
        "model": MODEL,
        "max_tokens": 16,
        "system": [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": f"Say hi briefly (call {call_idx})."}],
    }


def _auth_headers() -> dict[str, str]:
    """OAuth tokens (``sk-ant-oat...``) go in Authorization Bearer; API keys
    (``sk-ant-api...``) go in x-api-key. Anthropic rejects the wrong header
    with 401, so we route by prefix."""
    key = API_KEY or ""
    if key.startswith("sk-ant-oat"):
        return {
            "Authorization": f"Bearer {key}",
            "anthropic-beta": "oauth-2025-04-20,prompt-caching-2024-07-31",
        }
    return {
        "x-api-key": key,
        "anthropic-beta": "prompt-caching-2024-07-31",
    }


async def _call(client: httpx.AsyncClient, base_url: str, call_idx: int) -> dict[str, int]:
    r = await client.post(
        f"{base_url}/v1/messages",
        headers={
            **_auth_headers(),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json=_build_body(call_idx),
        timeout=120,
    )
    r.raise_for_status()
    usage = r.json()["usage"]
    return {
        "input": int(usage.get("input_tokens", 0)),
        "output": int(usage.get("output_tokens", 0)),
        "cache_creation": int(usage.get("cache_creation_input_tokens", 0)),
        "cache_read": int(usage.get("cache_read_input_tokens", 0)),
    }


def _verdict(first: dict[str, int], second: dict[str, int]) -> str:
    """A pass succeeds iff call 2 hits the cache.

    Whether call 1 wrote vs. read is incidental: if a prior probe run (or
    the other pass in the same run) already populated the cache for the
    same prefix, call 1 may read instead of write. The signal we care
    about is "the same prefix sent twice in this pass produces a cache
    hit on the second send."
    """
    if second["cache_read"] > 0:
        if first["cache_creation"] > 0:
            return "PASS  wrote on call 1, hit on call 2"
        if first["cache_read"] > 0:
            return "PASS  hit on both calls (cache pre-warmed by a prior run)"
        return "PASS  hit on call 2 (call 1 had no cache activity, surprising)"
    if first["cache_creation"] > 0:
        return "FAIL  wrote on call 1 but did NOT hit on call 2 (prefix changed?)"
    return "FAIL  no cache activity (prefix below model's min cacheable tokens?)"


def _fmt(label: str, u: dict[str, int]) -> str:
    return (
        f"  {label}: input={u['input']:>5} output={u['output']:>3} "
        f"cache_creation={u['cache_creation']:>5} cache_read={u['cache_read']:>5}"
    )


async def _probe(label: str, base_url: str) -> tuple[dict[str, int], dict[str, int]] | None:
    print(f"\n=== {label} ({base_url}) ===")
    async with httpx.AsyncClient() as client:
        try:
            first = await _call(client, base_url, 1)
            await asyncio.sleep(3.0)  # let Anthropic settle the cache write
            second = await _call(client, base_url, 2)
        except httpx.HTTPStatusError as exc:
            print(f"  ERROR {exc.response.status_code}: {exc.response.text[:300]}")
            return None
        except httpx.HTTPError as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            return None
    print(_fmt("call 1", first))
    print(_fmt("call 2", second))
    print(f"  -> {_verdict(first, second)}")
    return first, second


async def main() -> None:
    if not API_KEY:
        print("ANTHROPIC_API_KEY is required")
        sys.exit(2)
    print(f"model: {MODEL}")
    print(f"system prompt length: {len(SYSTEM_PROMPT)} chars")
    # Direct first so its baseline isn't polluted by magos's cache write.
    direct = await _probe("direct to anthropic (control)", ANTHROPIC_URL)
    via_magos = await _probe("through magos", MAGOS_URL)
    if direct and via_magos:
        d_tokens = direct[1]["cache_read"]
        m_tokens = via_magos[1]["cache_read"]
        print()
        if d_tokens and m_tokens and d_tokens == m_tokens:
            print(
                f"cross-pass: both passes hit the same {d_tokens}-token cache slot "
                f"=> compress preserved the prefix as upstream sees it"
            )
        elif d_tokens and m_tokens:
            print(
                f"cross-pass: cache_read differs (direct={d_tokens}, magos={m_tokens}) "
                f"=> the magos pass landed on a different cached prefix"
            )
        else:
            print("cross-pass: cannot compare (one side had no cache_read)")


if __name__ == "__main__":
    asyncio.run(main())
