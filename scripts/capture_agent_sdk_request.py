"""Capture the full HTTP request the Claude CLI sends to ANTHROPIC_BASE_URL.

Spins up a pure echo server that records the first incoming request, points
the agent SDK at it via ANTHROPIC_BASE_URL, runs a tiny query, and prints
method + path + headers + body of what Claude CLI actually sent.

Auth is partially redacted so the prefix structure is visible without
leaking the full bearer token.

Run with::

    uv run python scripts/capture_agent_sdk_request.py
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import threading
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, Request, Response

_BEARER_PARTS = 2
_REDACT_MIN_LEN = 16

captured: list[dict[str, Any]] = []


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _redact_auth(value: str) -> str:
    parts = value.split(" ", 1)
    if len(parts) == _BEARER_PARTS and parts[0].lower() == "bearer":
        token = parts[1]
        if len(token) > _REDACT_MIN_LEN:
            return f"Bearer {token[:8]}...{token[-4:]} (len={len(token)})"
        return f"Bearer <redacted len={len(token)}>"
    if len(value) > _REDACT_MIN_LEN:
        return f"{value[:8]}...{value[-4:]}"
    return "<redacted>"


def _build_app() -> FastAPI:
    app = FastAPI()

    @app.post("/{path:path}")
    async def echo(path: str, request: Request) -> Response:
        body_bytes = await request.body()
        headers_redacted: dict[str, str] = {}
        for k, v in request.headers.items():
            if k.lower() in {"authorization", "x-api-key", "cookie"}:
                headers_redacted[k] = _redact_auth(v)
            else:
                headers_redacted[k] = v
        captured.append(
            {
                "method": request.method,
                "path": "/" + path,
                "headers": headers_redacted,
                "body_size": len(body_bytes),
                "body_preview": body_bytes[:1500].decode("utf-8", errors="replace"),
            }
        )
        # Return a 401 so the SDK gives up quickly rather than retrying.
        return Response(
            content=b'{"type":"error","error":{"type":"capture_done","message":"captured"}}',
            status_code=401,
            media_type="application/json",
        )

    return app


def _start_server(port: int) -> tuple[uvicorn.Server, threading.Thread]:
    config = uvicorn.Config(
        _build_app(),
        host="127.0.0.1",
        port=port,
        log_level="error",
        log_config=None,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    return server, thread


async def _drive_agent_sdk(base_url: str) -> None:
    os.environ["ANTHROPIC_BASE_URL"] = base_url
    from claude_agent_sdk import ClaudeAgentOptions, query  # noqa: PLC0415

    async def _run() -> None:
        try:
            async for _ in query(
                prompt="hi",
                options=ClaudeAgentOptions(model="sonnet", allowed_tools=[]),
            ):
                pass
        except Exception as exc:
            print(f"(SDK exited: {type(exc).__name__}: {exc})")

    try:
        # Hard-cap so a 401-retry loop in the SDK does not hang the script.
        await asyncio.wait_for(_run(), timeout=20.0)
    except TimeoutError:
        print("(SDK timed out after 20s; capturing what we have)")


def main() -> None:
    port = _free_port()
    server, thread = _start_server(port)
    try:
        asyncio.run(_drive_agent_sdk(f"http://127.0.0.1:{port}"))
    finally:
        server.should_exit = True
        thread.join(timeout=3)

    if not captured:
        print("NO REQUESTS CAPTURED — agent SDK never reached the echo server")
        return

    for i, req in enumerate(captured):
        print(f"\n=== request {i + 1}/{len(captured)} ===")
        print(f"{req['method']} {req['path']}")
        print(f"body_size: {req['body_size']} bytes")
        print("headers:")
        for k, v in req["headers"].items():
            print(f"  {k}: {v}")
        print("body_preview:")
        try:
            parsed = json.loads(req["body_preview"])
            print(json.dumps(parsed, indent=2)[:1500])
        except Exception:
            print(req["body_preview"])


if __name__ == "__main__":
    main()
