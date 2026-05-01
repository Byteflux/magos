"""Capture what magos forwards upstream when driven by the agent SDK.

Spins up two servers:

1. **Echo** (port E): records every incoming POST and returns 401.
2. **Magos** (port M): real magos uvicorn, with anthropic_upstream_url
   pointed at ``http://127.0.0.1:{E}``.

Then runs the agent SDK with ``ANTHROPIC_BASE_URL=http://127.0.0.1:{M}`` so
the call path is ``Claude CLI -> magos -> echo``. Comparing the captured
echo request against the direct-capture (``capture_agent_sdk_request.py``)
reveals exactly what magos changes about the request, if anything.
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

captured: list[dict[str, Any]] = []


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


_BEARER_PARTS = 2
_REDACT_MIN_LEN = 16


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


def _build_echo_app() -> FastAPI:
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
        return Response(
            content=b'{"type":"error","error":{"type":"capture_done","message":"captured"}}',
            status_code=401,
            media_type="application/json",
        )

    return app


def _start_server(app: FastAPI | str, port: int) -> tuple[uvicorn.Server, threading.Thread]:
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", log_config=None)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    return server, thread


async def _drive_agent_sdk(magos_base: str) -> None:
    os.environ["ANTHROPIC_BASE_URL"] = magos_base
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
        await asyncio.wait_for(_run(), timeout=20.0)
    except TimeoutError:
        print("(SDK timed out after 20s)")


def main() -> None:
    echo_port = _free_port()
    magos_port = _free_port()

    # Magos must read the upstream URL from MagosSettings; set the env var
    # BEFORE importing/starting the server.
    os.environ["MAGOS_ANTHROPIC_UPSTREAM_URL"] = f"http://127.0.0.1:{echo_port}"
    os.environ["MAGOS_ANTHROPIC_PASSTHROUGH_ENABLED"] = "true"

    echo_server, echo_thread = _start_server(_build_echo_app(), echo_port)
    magos_server, magos_thread = _start_server("magos.server:app", magos_port)
    try:
        asyncio.run(_drive_agent_sdk(f"http://127.0.0.1:{magos_port}"))
    finally:
        echo_server.should_exit = True
        magos_server.should_exit = True
        echo_thread.join(timeout=3)
        magos_thread.join(timeout=3)

    if not captured:
        print("NO REQUESTS CAPTURED at echo")
        return

    for i, req in enumerate(captured):
        print(f"\n=== request {i + 1}/{len(captured)} (echo received from magos) ===")
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
