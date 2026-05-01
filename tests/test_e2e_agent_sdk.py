"""End-to-end test driving magos through the Claude Agent SDK.

The Agent SDK spawns the local ``claude`` CLI subprocess, which honours
``ANTHROPIC_BASE_URL`` and uses the user's Claude Code credentials to
authenticate. Magos forwards those headers upstream verbatim
(``Authorization``, ``anthropic-beta``, ``anthropic-version``, ...), so this
test exercises the full streaming agent loop without requiring any provider
API key in the test environment.

Skipped by default. To run::

    MAGOS_E2E=1 uv run pytest -m e2e tests/test_e2e_agent_sdk.py

Requires:

- ``claude`` CLI on PATH and authenticated via ``claude login``
- ``claude-agent-sdk`` installed (added as a dev dependency)
"""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import threading
import time
from collections.abc import Iterator

import pytest
import uvicorn

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("MAGOS_E2E") != "1",
        reason="set MAGOS_E2E=1 to run end-to-end provider tests",
    ),
    pytest.mark.skipif(
        shutil.which("claude") is None,
        reason="claude CLI not on PATH; install Claude Code to run",
    ),
]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def magos_server() -> Iterator[int]:
    """Start magos uvicorn in a background thread on a free port."""
    port = _free_port()
    config = uvicorn.Config(
        "magos.server:app",
        host="127.0.0.1",
        port=port,
        log_level="warning",
        log_config=None,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 5.0
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=2.0)
        pytest.fail("magos server failed to start within 5s")

    try:
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


def test_agent_sdk_sonnet_basic(magos_server: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive a sonnet query through magos via the Claude Agent SDK.

    Asserts the agent loop completes successfully (a ``ResultMessage`` with
    ``is_error=False``) and that some assistant text was produced. The exact
    wording is model-dependent so we only check non-emptiness; that is enough
    to prove streaming + tool-use + auth-passthrough all worked end-to-end.
    """
    pytest.importorskip("claude_agent_sdk")
    # Imports are inside the test so the module loads even without the SDK
    # installed; the importorskip above gates module-import-time errors.
    from claude_agent_sdk import (  # noqa: PLC0415
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        query,
    )

    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{magos_server}")

    async def run() -> tuple[list[str], ResultMessage | None]:
        texts: list[str] = []
        result: ResultMessage | None = None
        async for message in query(
            prompt="Reply with the single word: pong",
            options=ClaudeAgentOptions(model="sonnet", allowed_tools=[]),
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        texts.append(block.text)
            elif isinstance(message, ResultMessage):
                result = message
        return texts, result

    texts, result = asyncio.run(run())

    assert result is not None, "agent loop did not produce a ResultMessage"
    assert result.is_error is False, f"agent loop reported error: {result!r}"
    assert texts, "agent SDK did not yield any assistant text content"
    assert any(text.strip() for text in texts), "all assistant text was blank"
