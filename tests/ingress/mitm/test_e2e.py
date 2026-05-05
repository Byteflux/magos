"""End-to-end test for the embedded mitmproxy HTTPS_PROXY ingress.

Spawns ``magos serve`` in a subprocess with ``MAGOS_HOME`` anchored at the
project root, points an httpx client at the mitm listener via
``HTTPS_PROXY=http://127.0.0.1:<port>``, and asserts the request reaches
the routing layer through TLS termination + decrypted-request rewrite.

The check uses a model id no shipped rule matches, so the request hits
magos's routing 404 envelope without any upstream call. That's enough to
prove the full ingress path: TLS handshake against the addon-signed leaf
cert, ``tls_clienthello`` allowlist gate, ``request`` host/port/scheme
rewrite, FastAPI routing, response stream back through mitmproxy.

Skipped by default; gate matches the rest of the e2e suite::

    MAGOS_E2E=1 uv run pytest tests/ingress/mitm/test_e2e.py
"""

from __future__ import annotations

import os
import socket
import ssl
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("MAGOS_E2E") != "1",
        reason="set MAGOS_E2E=1 to run end-to-end ingress tests",
    ),
]

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MITM_CA = Path.home() / ".mitmproxy" / "mitmproxy-ca-cert.pem"
# Wait for both: the ingress listener bound and the Kompress weights
# loaded. Kompress holds an internal threading.Lock during the download;
# a request that hits the compress pre-rewrite while the preload is
# still in flight blocks the asyncio loop and the connection eventually
# drops with a RemoteProtocolError. Gating on the warmup event is the
# only reliable way to keep the test deterministic on a cold cache.
_STARTUP_LOG_LINES = ("ingress.started", "compress.kompress_warmed")
_STARTUP_TIMEOUT_SECONDS = 120.0


def _free_port() -> int:
    """Bind to port 0, read the OS-assigned port, release the socket."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


@pytest.fixture
def magos_serve(tmp_path: Path) -> Iterator[tuple[int, int, Path]]:
    """Spawn ``magos serve`` with mitm ingress on free ports; yield (http, mitm, log).

    ``MAGOS_HOME`` is pinned to the project root so the subprocess uses
    the project's ``magos.yaml`` and ``models.json`` rather than the
    operator's ``~/.magos/`` (which may point at a Docker-mounted config
    or other state unrelated to this test).
    """
    if not _MITM_CA.is_file():
        pytest.skip(f"mitmproxy CA not found at {_MITM_CA}; run mitmdump once to seed it")

    http_port = _free_port()
    mitm_port = _free_port()
    log_path = tmp_path / "magos-serve.log"

    env = os.environ.copy()
    env.update(
        {
            "MAGOS_HOME": str(_PROJECT_ROOT),
            "MAGOS_HOST": "127.0.0.1",
            "MAGOS_PORT": str(http_port),
            "MAGOS_MITM_ENABLED": "true",
            "MAGOS_MITM_HOST": "127.0.0.1",
            "MAGOS_MITM_PORT": str(mitm_port),
            "MAGOS_MITM_INTERCEPT_HOSTS": "api.anthropic.com",
            "MAGOS_LOG_JSON": "0",
        },
    )

    with log_path.open("wb") as logf:
        proc = subprocess.Popen(
            [sys.executable, "-m", "magos", "serve"],
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(_PROJECT_ROOT),
        )

        deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
        ready = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"magos serve exited early (code {proc.returncode}); see {log_path}"
                )
            text = log_path.read_text(errors="replace")
            if all(line in text for line in _STARTUP_LOG_LINES):
                ready = True
                break
            time.sleep(0.2)
        if not ready:
            proc.terminate()
            raise TimeoutError(
                f"magos serve never logged all of {_STARTUP_LOG_LINES} within "
                f"{_STARTUP_TIMEOUT_SECONDS}s; see {log_path}"
            )

        try:
            yield http_port, mitm_port, log_path
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def test_https_proxy_routes_to_fastapi(magos_serve: tuple[int, int, Path]) -> None:
    """HTTPS_PROXY -> mitmproxy -> FastAPI -> routing 404 envelope.

    Uses an unmatched model id so no upstream call is made; the test
    proves the ingress path end-to-end without external dependencies.
    """
    _, mitm_port, _ = magos_serve

    body = {
        "model": "no-such-model-anywhere",
        "max_tokens": 4,
        "messages": [{"role": "user", "content": "hi"}],
    }
    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        proxy=f"http://127.0.0.1:{mitm_port}",
        verify=ssl.create_default_context(cafile=str(_MITM_CA)),
        timeout=15,
        json=body,
        headers={"anthropic-version": "2023-06-01"},
    )

    assert resp.status_code == 404, resp.text
    payload = resp.json()
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "not_found_error"
    assert "no-such-model-anywhere" in payload["error"]["message"]


def test_ingress_rewrite_logged(magos_serve: tuple[int, int, Path]) -> None:
    """The ``ingress.rewrote`` event fires when the addon rewrites a request.

    Sanity-checks the addon's ``request`` hook actually ran (rather than
    the request having reached FastAPI by some other path), since the
    rewrite is the load-bearing step that makes the proxy mode work.
    """
    _, mitm_port, log_path = magos_serve

    httpx.post(
        "https://api.anthropic.com/v1/messages",
        proxy=f"http://127.0.0.1:{mitm_port}",
        verify=ssl.create_default_context(cafile=str(_MITM_CA)),
        timeout=15,
        json={
            "model": "no-such-model-anywhere",
            "max_tokens": 4,
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={"anthropic-version": "2023-06-01"},
    )

    # Allow a beat for the structlog line to flush to disk.
    deadline = time.monotonic() + 5.0
    text = ""
    while time.monotonic() < deadline:
        text = log_path.read_text(errors="replace")
        if "ingress.rewrote" in text:
            return
        time.sleep(0.1)
    raise AssertionError(f"expected 'ingress.rewrote' in magos log; last contents:\n{text[-2000:]}")


def test_ca_mismatch_documents_failure_mode(magos_serve: tuple[int, int, Path]) -> None:
    """When the client trusts a CA other than the one signing leaf certs, the
    handshake fails. Documents the Docker-container case where the host's
    ``~/.mitmproxy/`` is not mounted into the container, so the container
    generates its own CA on first run and the host's trust store can never
    verify it. The fix is to mount ``~/.mitmproxy:/root/.mitmproxy`` (see
    docs/deployment.md), not anything in magos itself.
    """
    _, mitm_port, _ = magos_serve

    # Trust nothing: emulates the Docker case where the host CA doesn't
    # match the proxy's signing CA. Any unrelated CA bundle would fail
    # the same way; ``verify=True`` against the system store is the
    # cleanest stand-in.
    with pytest.raises(httpx.ConnectError, match="CERTIFICATE_VERIFY_FAILED"):
        httpx.post(
            "https://api.anthropic.com/v1/messages",
            proxy=f"http://127.0.0.1:{mitm_port}",
            verify=True,
            timeout=15,
            json={"model": "claude-x", "max_tokens": 1, "messages": []},
        )
