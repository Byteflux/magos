"""``KompressBackendOverride``: ``MAGOS_KOMPRESS_BACKEND=pytorch`` flips
Headroom's ONNX-availability check so the loader takes the PyTorch branch.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from magos.api import create_app
from tests.api._helpers import translate_only_cfg

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
