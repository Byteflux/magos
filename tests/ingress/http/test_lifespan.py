"""Lifespan tests: magos.compression warmup and kompress-backend override."""

from __future__ import annotations

import importlib
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

import magos.compression as mc
from magos.compression import PipelineConfig
from magos.compression import registry as reg_mod
from magos.ingress.http import create_app
from magos.routing import RoutingConfig

from ._helpers import translate_only_cfg

# --- Lifespan: magos.compression warmup ---


def test_lifespan_warms_compress_pipeline_when_rule_uses_compress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If any rule has a Compress rewrite, startup must build pipelines + run eager_warmup."""
    built: list[tuple[str, str]] = []
    eager_calls: list[int] = []

    def fake_get_or_build(self: object, config: PipelineConfig, *, provider_name: str) -> object:
        built.append((config.fingerprint(), provider_name))
        return object()

    def fake_eager_warmup(_registry: object) -> None:
        eager_calls.append(1)

    monkeypatch.setattr(reg_mod.PipelineRegistry, "get_or_build", fake_get_or_build)
    # ``prebuild_from_routing`` (called by the lifespan) calls ``eager_warmup``
    # via its own module-local reference, so patch at the call site.
    from magos.compression import warmup as warmup_mod  # noqa: PLC0415

    monkeypatch.setattr(warmup_mod, "eager_warmup", fake_eager_warmup)

    cfg = RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/messages"}},
                    "rewrites": [{"compress": {}}],
                    "action": {"provider": "anthropic", "mode": "translate"},
                }
            ]
        }
    )
    app = create_app(routing=cfg)
    with TestClient(app):
        pass

    fp = PipelineConfig().fingerprint()
    assert (fp, "anthropic") in built
    assert (fp, "openai") in built
    assert eager_calls == [1]


def test_lifespan_skips_warmup_when_no_compress_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No Compress rewrite anywhere -> never touch the magos.compression registry."""
    built: list[tuple[str, str]] = []

    def fake_get_or_build(self: object, config: PipelineConfig, *, provider_name: str) -> object:
        built.append((config.fingerprint(), provider_name))
        return object()

    eager_calls: list[int] = []

    monkeypatch.setattr(reg_mod.PipelineRegistry, "get_or_build", fake_get_or_build)
    monkeypatch.setattr(mc, "eager_warmup", lambda _r: eager_calls.append(1))

    cfg = translate_only_cfg()
    app = create_app(routing=cfg)
    with TestClient(app):
        pass

    assert built == []
    assert eager_calls == []


def test_lifespan_warmup_failure_does_not_block_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken pipeline build must log + continue, not crash the app."""

    def boom(self: object, *args: object, **kwargs: object) -> object:
        raise RuntimeError("pipeline init failed")

    monkeypatch.setattr(reg_mod.PipelineRegistry, "get_or_build", boom)

    cfg = RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/messages"}},
                    "rewrites": [{"compress": {}}],
                    "action": {"provider": "anthropic", "mode": "translate"},
                }
            ]
        }
    )
    app = create_app(routing=cfg)
    with TestClient(app) as client:
        resp = client.post("/v1/messages", json={"model": "x", "messages": []})
    assert resp.status_code != 500


# --- Lifespan: kompress_backend override ---


# Capture the real Kompress ONNX availability check at module import time,
# before any test or lifespan can replace it. The override-test pair below
# reset to this baseline at the start of each run so they're robust to
# external env state (e.g. running the suite with MAGOS_KOMPRESS_BACKEND
# already exported).
_kc_module = importlib.import_module("headroom.transforms.kompress_compressor")
_KC_ORIGINAL_IS_ONNX_AVAILABLE = _kc_module._is_onnx_available


@pytest.fixture
def _restore_kompress_onnx_check() -> Iterator[None]:
    """Restore the real ONNX-availability check around the test."""
    _kc_module._is_onnx_available = _KC_ORIGINAL_IS_ONNX_AVAILABLE  # type: ignore[attr-defined]
    try:
        yield
    finally:
        _kc_module._is_onnx_available = _KC_ORIGINAL_IS_ONNX_AVAILABLE  # type: ignore[attr-defined]


def test_lifespan_forces_pytorch_when_kompress_backend_set(
    monkeypatch: pytest.MonkeyPatch,
    _restore_kompress_onnx_check: None,
) -> None:
    """``MAGOS_KOMPRESS_BACKEND=pytorch`` flips _is_onnx_available to False
    so Headroom's loader takes the PyTorch branch on first compress call.
    """
    monkeypatch.setenv("MAGOS_KOMPRESS_BACKEND", "pytorch")
    # Pre-condition: with onnxruntime + transformers installed, this is True.
    assert _kc_module._is_onnx_available() is True

    cfg = translate_only_cfg()
    app = create_app(routing=cfg)
    with TestClient(app):
        pass

    # After lifespan ran with backend=pytorch, the module-level binding is
    # the False-returning stub.
    assert _kc_module._is_onnx_available() is False


def test_lifespan_default_leaves_onnx_check_untouched(
    monkeypatch: pytest.MonkeyPatch,
    _restore_kompress_onnx_check: None,
) -> None:
    """Default (auto) backend must not patch the ONNX availability check."""
    monkeypatch.delenv("MAGOS_KOMPRESS_BACKEND", raising=False)

    cfg = translate_only_cfg()
    app = create_app(routing=cfg)
    with TestClient(app):
        pass

    # Function identity preserved: no monkeypatch by lifespan.
    assert _kc_module._is_onnx_available is _KC_ORIGINAL_IS_ONNX_AVAILABLE
