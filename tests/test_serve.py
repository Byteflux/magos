"""Tests for the process orchestrator (FastAPI + optional mitm task).

Covers the env-over-yaml bind layering and that the mitm task is only
spawned when explicitly enabled with at least one intercept host.
The real ``DumpMaster`` and ``uvicorn.Server`` are mocked out — full
network integration belongs in manual smoke testing per the plan.
"""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path
from typing import Any

import pytest

from magos.config.schema import IngressConfig, MagosServerConfig
from magos.config.settings import MagosSettings
from magos.serve import resolve_bind, serve_async


@pytest.mark.unit
def test_resolve_bind_yaml_default_when_env_unset() -> None:
    settings = MagosSettings(_env_file=None)  # type: ignore[call-arg]
    server_cfg = MagosServerConfig(host="0.0.0.0", port=9000)
    assert resolve_bind(settings, server_cfg) == ("0.0.0.0", 9000)


@pytest.mark.unit
def test_resolve_bind_env_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAGOS_HOST", "10.0.0.1")
    monkeypatch.setenv("MAGOS_PORT", "7000")
    settings = MagosSettings(_env_file=None)  # type: ignore[call-arg]
    server_cfg = MagosServerConfig(host="0.0.0.0", port=9000)
    assert resolve_bind(settings, server_cfg) == ("10.0.0.1", 7000)


@pytest.mark.unit
def test_resolve_bind_empty_env_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty MAGOS_HOST shouldn't shadow a real yaml default."""
    monkeypatch.setenv("MAGOS_HOST", "")
    monkeypatch.delenv("MAGOS_PORT", raising=False)
    settings = MagosSettings(_env_file=None)  # type: ignore[call-arg]
    server_cfg = MagosServerConfig(host="0.0.0.0")
    host, _ = resolve_bind(settings, server_cfg)
    assert host == "0.0.0.0"


class _StubServer:
    """Stand-in for ``uvicorn.Server`` that completes immediately."""

    def __init__(self, *, fail: bool = False, lifespan_delay: float = 0.0) -> None:
        self.started = False
        self.should_exit = False
        self._fail = fail
        self._lifespan_delay = lifespan_delay
        self.serve_called = False

    async def serve(self) -> None:
        self.serve_called = True
        if self._lifespan_delay:
            await asyncio.sleep(self._lifespan_delay)
        self.started = True
        if self._fail:
            raise RuntimeError("simulated FastAPI startup failure")
        # Stay alive until told to exit (or until cancelled by orchestrator).
        while not self.should_exit:  # noqa: ASYNC110
            await asyncio.sleep(0.01)


class _StubMaster:
    """Stand-in for mitmproxy ``DumpMaster``."""

    def __init__(self) -> None:
        self._stopping = asyncio.Event()
        self.run_called = False
        self.shutdown_called = False

    async def run(self) -> None:
        self.run_called = True
        await self._stopping.wait()

    def shutdown(self) -> None:
        self.shutdown_called = True
        self._stopping.set()


def _stub_create_app(**_kwargs: Any) -> object:
    return object()


_FIXTURE_YAML = Path(__file__).parent / "fixtures" / "magos.test.yaml"


@pytest.fixture
def patched_orchestrator(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch the orchestrator's collaborators with stand-ins."""
    # Capture the real loader BEFORE monkeypatch so our stub can reuse it.
    from magos.config.loader import load_full_config as real_load_full_config  # noqa: PLC0415

    state: dict[str, Any] = {"server_cfg": MagosServerConfig()}

    def fake_create_app(**kwargs: Any) -> object:
        state["create_app_kwargs"] = kwargs
        return object()

    def fake_uvi_config(*args: Any, **kwargs: Any) -> Any:
        state["uvi_config_kwargs"] = kwargs
        return object()

    server = _StubServer()
    master = _StubMaster()
    state["server"] = server
    state["master"] = master

    def fake_uvi_server(_config: Any) -> _StubServer:
        return server

    def fake_build_master(*_args: Any, **kwargs: Any) -> _StubMaster:
        state["build_master_kwargs"] = kwargs
        return master

    def fake_install_bridge() -> None:
        state["install_bridge_called"] = True

    def fake_load(_path: str | Path):  # type: ignore[no-untyped-def]
        real = real_load_full_config(_FIXTURE_YAML)
        return dataclasses.replace(real, server=state["server_cfg"])

    monkeypatch.setattr("magos.serve.create_app", fake_create_app)
    monkeypatch.setattr("magos.serve.uvicorn.Config", fake_uvi_config)
    monkeypatch.setattr("magos.serve.uvicorn.Server", fake_uvi_server)
    monkeypatch.setattr("magos.serve.build_ingress_master", fake_build_master)
    monkeypatch.setattr("magos.serve.install_log_bridge", fake_install_bridge)
    # serve.py captures load_full_config at import time, so patch the
    # name on the serve module itself rather than at the source.
    monkeypatch.setattr("magos.serve.load_full_config", fake_load)
    return state


@pytest.mark.unit
def test_orchestrator_skips_ingress_when_disabled(
    patched_orchestrator: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    patched_orchestrator["server_cfg"] = MagosServerConfig(ingress=IngressConfig(enabled=False))
    settings = MagosSettings(_env_file=None)  # type: ignore[call-arg]

    async def runner() -> None:
        task = asyncio.create_task(serve_async(settings=settings))
        # Let the server start, then trigger shutdown.
        await asyncio.sleep(0.05)
        patched_orchestrator["server"].should_exit = True
        await asyncio.wait_for(task, timeout=2.0)

    asyncio.run(runner())

    assert patched_orchestrator["server"].serve_called is True
    assert patched_orchestrator["master"].run_called is False
    assert "install_bridge_called" not in patched_orchestrator


@pytest.mark.unit
def test_orchestrator_skips_ingress_when_no_intercept_hosts(
    patched_orchestrator: dict[str, Any],
) -> None:
    patched_orchestrator["server_cfg"] = MagosServerConfig(
        ingress=IngressConfig(enabled=True, intercept_hosts=())
    )
    settings = MagosSettings(_env_file=None)  # type: ignore[call-arg]

    async def runner() -> None:
        task = asyncio.create_task(serve_async(settings=settings))
        await asyncio.sleep(0.05)
        patched_orchestrator["server"].should_exit = True
        await asyncio.wait_for(task, timeout=2.0)

    asyncio.run(runner())
    # Empty allowlist => warn & skip; mitm never starts even though enabled.
    assert patched_orchestrator["master"].run_called is False


@pytest.mark.unit
def test_orchestrator_starts_both_when_enabled(patched_orchestrator: dict[str, Any]) -> None:
    patched_orchestrator["server_cfg"] = MagosServerConfig(
        ingress=IngressConfig(
            enabled=True,
            intercept_hosts=("api.anthropic.com",),
        )
    )
    settings = MagosSettings(_env_file=None)  # type: ignore[call-arg]

    async def runner() -> None:
        task = asyncio.create_task(serve_async(settings=settings))
        # Allow time for FastAPI startup-poll + mitm task spawn.
        await asyncio.sleep(0.2)
        # Bring FastAPI down -> orchestrator should shut mitm and return.
        patched_orchestrator["server"].should_exit = True
        await asyncio.wait_for(task, timeout=2.0)

    asyncio.run(runner())

    server = patched_orchestrator["server"]
    master = patched_orchestrator["master"]
    assert server.serve_called is True
    assert master.run_called is True
    assert master.shutdown_called is True
    assert patched_orchestrator["install_bridge_called"] is True
    # Bind layering propagated through to the master factory.
    kwargs = patched_orchestrator["build_master_kwargs"]
    assert kwargs["target_host"] == "127.0.0.1"
    assert kwargs["target_port"] == 8000
