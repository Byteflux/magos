"""Smoke tests for ``magos.telemetry.tracing``.

The ``traced`` decorator must preserve return values and propagate
exceptions whether or not OTel is configured. ``configure_tracing`` is
a no-op when ``MAGOS_OTEL_ENABLED`` is unset.
"""

from __future__ import annotations

import pytest

from magos.telemetry import configure_tracing, traced


@pytest.mark.unit
def test_traced_preserves_return_value() -> None:
    @traced()
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5


@pytest.mark.unit
def test_traced_propagates_exceptions() -> None:
    @traced("failing-op")
    def boom() -> None:
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        boom()


@pytest.mark.unit
def test_configure_tracing_is_noop_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MAGOS_OTEL_ENABLED", raising=False)
    configure_tracing()
