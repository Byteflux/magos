"""``MagosCompressionWarmup``: builds pipelines for both providers and runs
eager warmup; failures are non-fatal.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import magos.compression as mc
from magos.api import create_app
from magos.compression import PipelineConfig
from magos.compression import registry as reg_mod
from magos.routing import RoutingConfig
from tests.api._helpers import translate_only_cfg


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
                    "transforms": [{"compress": {}}],
                    "target": {"provider": "anthropic", "gateway": "translate"},
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
                    "transforms": [{"compress": {}}],
                    "target": {"provider": "anthropic", "gateway": "translate"},
                }
            ]
        }
    )
    app = create_app(routing=cfg)
    with TestClient(app) as client:
        resp = client.post("/v1/messages", json={"model": "x", "messages": []})
    assert resp.status_code != 500
