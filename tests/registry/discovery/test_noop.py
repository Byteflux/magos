"""``NoopAdapter`` returns no models regardless of upstream state."""

from __future__ import annotations

from magos.registry.discovery.noop import NoopAdapter
from magos.registry.schema import ProviderConfig
from tests.registry.discovery._helpers import err, run_discover


def test_noop_adapter_returns_empty_result() -> None:
    cfg = ProviderConfig.model_validate({})
    # Even if the upstream is broken (500), noop never reaches it; the
    # adapter exists so manual-only providers can be wired without a
    # discovery path.
    result = run_discover(NoopAdapter(), "manual", cfg, err(500))
    assert result.models == ()
