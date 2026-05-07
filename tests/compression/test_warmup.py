"""``eager_warmup`` calls ``eager_load_compressors`` once per unique transform."""

from __future__ import annotations

from typing import Any

from magos.compression import get_registry
from magos.compression.registry import PipelineRegistry
from magos.compression.warmup import eager_warmup


class _EagerTransform:
    def __init__(self) -> None:
        self.calls = 0

    def eager_load_compressors(self) -> dict[str, Any]:
        self.calls += 1
        return {"loaded": True}


class _PlainTransform:
    pass


class _StubPipeline:
    def __init__(self, transforms: list[Any]) -> None:
        self.transforms = transforms


class _StubRegistry(PipelineRegistry):
    """Registry stub: returns the pipelines passed to its constructor.

    Subclassing skips PipelineRegistry.__init__ — we don't want a cache.
    """

    def __init__(self, pipelines: list[_StubPipeline]) -> None:
        self._pipelines = pipelines

    def pipelines(self) -> Any:
        return iter(self._pipelines)


def test_warmup_invokes_eager_load_on_each_unique_transform() -> None:
    eager = _EagerTransform()
    plain = _PlainTransform()
    reg = _StubRegistry([_StubPipeline([eager, plain]), _StubPipeline([eager, plain])])

    eager_warmup(reg)

    assert eager.calls == 1


def test_warmup_swallows_eager_load_errors() -> None:
    class _Boom:
        def eager_load_compressors(self) -> None:
            raise RuntimeError("boom")

    boom = _Boom()
    reg = _StubRegistry([_StubPipeline([boom])])

    eager_warmup(reg)  # must not raise


def test_warmup_default_uses_module_registry() -> None:
    eager_warmup()  # smoke: must not raise on empty registry
    assert isinstance(get_registry(), PipelineRegistry)


def _make_routing_config(
    rule_compress_options: list[Any] | None = None,
    pre_rewrites_compress_options: list[Any] | None = None,
    guarded_pre_compress_options: list[Any] | None = None,
) -> Any:
    """Build a RoutingConfig with the supplied Compress options arrangement.

    - rule_compress_options: each entry becomes a rule with that one Compress.
    - pre_rewrites_compress_options: each entry becomes a top-level pre_rewrite Compress.
    - guarded_pre_compress_options: each entry becomes a Compress inside a single
      GuardedRewrites pre_rewrite (all share one match).
    """
    from magos.routing.schema import (  # noqa: PLC0415
        Compress,
        EndpointAtom,
        GuardedRewrites,
        LiteralMatcher,
        RoutingConfig,
        Rule,
        Target,
    )

    match = EndpointAtom(endpoint=LiteralMatcher(literal="/v1/messages"))
    rules = [
        Rule(
            match=match,
            rewrites=[Compress(compress=opts)],
            target=Target(provider="anthropic", gateway="passthrough", base_url="https://x"),
        )
        for opts in (rule_compress_options or [])
    ]
    if not rules:
        # RoutingConfig requires at least one rule; add an empty-rewrites one.
        rules = [
            Rule(
                match=match,
                rewrites=[],
                target=Target(provider="anthropic", gateway="passthrough", base_url="https://x"),
            )
        ]
    pre: list[Any] = [Compress(compress=opts) for opts in (pre_rewrites_compress_options or [])]
    if guarded_pre_compress_options:
        pre.append(
            GuardedRewrites(
                match=match,
                rewrites=[Compress(compress=opts) for opts in guarded_pre_compress_options],
            )
        )
    return RoutingConfig(pre_rewrites=pre, rules=rules)


def test_prebuild_from_routing_empty_config_builds_nothing(monkeypatch: Any) -> None:
    from magos.compression import prebuild_from_routing  # noqa: PLC0415
    from magos.compression.registry import PipelineRegistry  # noqa: PLC0415

    cfg = _make_routing_config()  # one rule, zero rewrites
    reg = PipelineRegistry()

    # Patch eager_warmup so we can confirm it was called once even with no builds.
    calls: list[Any] = []
    monkeypatch.setattr("magos.compression.warmup.eager_warmup", lambda r=None: calls.append(r))

    prebuild_from_routing(cfg, registry=reg)

    assert list(reg.pipelines()) == []
    assert calls == [reg]


def test_prebuild_from_routing_default_compress_builds_two_pipelines(
    monkeypatch: Any,
) -> None:
    """One rule with default Compress -> two pipelines (anthropic + openai)."""
    from magos.compression import prebuild_from_routing  # noqa: PLC0415
    from magos.compression.registry import PipelineRegistry  # noqa: PLC0415
    from magos.routing.schema import CompressOptions  # noqa: PLC0415

    cfg = _make_routing_config(rule_compress_options=[CompressOptions()])
    reg = PipelineRegistry()

    monkeypatch.setattr("magos.compression.warmup.eager_warmup", lambda r=None: None)

    prebuild_from_routing(cfg, registry=reg)

    assert len(list(reg.pipelines())) == 2


def test_prebuild_from_routing_dedups_repeated_configs(monkeypatch: Any) -> None:
    """Five rules using the same default options -> still 2 pipelines."""
    from magos.compression import prebuild_from_routing  # noqa: PLC0415
    from magos.compression.registry import PipelineRegistry  # noqa: PLC0415
    from magos.routing.schema import CompressOptions  # noqa: PLC0415

    cfg = _make_routing_config(rule_compress_options=[CompressOptions()] * 5)
    reg = PipelineRegistry()

    monkeypatch.setattr("magos.compression.warmup.eager_warmup", lambda r=None: None)

    prebuild_from_routing(cfg, registry=reg)

    assert len(list(reg.pipelines())) == 2


def test_prebuild_from_routing_distinct_configs_yield_distinct_pipelines(
    monkeypatch: Any,
) -> None:
    """Two rules with different smart_routing -> 4 pipelines (2 fingerprints x 2 providers)."""
    from magos.compression import prebuild_from_routing  # noqa: PLC0415
    from magos.compression.registry import PipelineRegistry  # noqa: PLC0415
    from magos.routing.schema import CompressOptions  # noqa: PLC0415

    cfg = _make_routing_config(
        rule_compress_options=[
            CompressOptions(smart_routing=True),
            CompressOptions(smart_routing=False),
        ]
    )
    reg = PipelineRegistry()

    monkeypatch.setattr("magos.compression.warmup.eager_warmup", lambda r=None: None)

    prebuild_from_routing(cfg, registry=reg)

    assert len(list(reg.pipelines())) == 4


def test_prebuild_from_routing_skips_cache_mode(monkeypatch: Any) -> None:
    """A cache-mode-only Compress -> no pipelines built."""
    from magos.compression import prebuild_from_routing  # noqa: PLC0415
    from magos.compression.registry import PipelineRegistry  # noqa: PLC0415
    from magos.routing.schema import CompressOptions  # noqa: PLC0415

    cfg = _make_routing_config(rule_compress_options=[CompressOptions(mode="cache")])
    reg = PipelineRegistry()

    monkeypatch.setattr("magos.compression.warmup.eager_warmup", lambda r=None: None)

    prebuild_from_routing(cfg, registry=reg)

    assert list(reg.pipelines()) == []


def test_prebuild_from_routing_walks_pre_rewrites(monkeypatch: Any) -> None:
    """A token-mode Compress in pre_rewrites is honored."""
    from magos.compression import prebuild_from_routing  # noqa: PLC0415
    from magos.compression.registry import PipelineRegistry  # noqa: PLC0415
    from magos.routing.schema import CompressOptions  # noqa: PLC0415

    cfg = _make_routing_config(pre_rewrites_compress_options=[CompressOptions()])
    reg = PipelineRegistry()

    monkeypatch.setattr("magos.compression.warmup.eager_warmup", lambda r=None: None)

    prebuild_from_routing(cfg, registry=reg)

    assert len(list(reg.pipelines())) == 2


def test_prebuild_from_routing_walks_guarded_pre_rewrites(monkeypatch: Any) -> None:
    """A token-mode Compress nested inside a GuardedRewrites pre_rewrite is honored."""
    from magos.compression import prebuild_from_routing  # noqa: PLC0415
    from magos.compression.registry import PipelineRegistry  # noqa: PLC0415
    from magos.routing.schema import CompressOptions  # noqa: PLC0415

    cfg = _make_routing_config(guarded_pre_compress_options=[CompressOptions()])
    reg = PipelineRegistry()

    monkeypatch.setattr("magos.compression.warmup.eager_warmup", lambda r=None: None)

    prebuild_from_routing(cfg, registry=reg)

    assert len(list(reg.pipelines())) == 2


def test_prebuild_from_routing_isolates_per_pipeline_failures(
    monkeypatch: Any,
) -> None:
    """One failing build_pipeline call doesn't stop the others."""
    from magos.compression import prebuild_from_routing  # noqa: PLC0415
    from magos.compression.registry import PipelineRegistry  # noqa: PLC0415
    from magos.routing.schema import CompressOptions  # noqa: PLC0415

    cfg = _make_routing_config(rule_compress_options=[CompressOptions()])

    class _BadRegistry(PipelineRegistry):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[str, str]] = []

        def get_or_build(self, config: Any, *, provider_name: Any) -> Any:
            self.calls.append((config.fingerprint(), provider_name))
            if provider_name == "anthropic":
                raise RuntimeError("synthetic build failure")
            return super().get_or_build(config, provider_name=provider_name)

    reg = _BadRegistry()
    monkeypatch.setattr("magos.compression.warmup.eager_warmup", lambda r=None: None)

    # Must not raise.
    prebuild_from_routing(cfg, registry=reg)

    providers_attempted = {p for _, p in reg.calls}
    assert providers_attempted == {"anthropic", "openai"}
    assert len(list(reg.pipelines())) == 1


def test_prebuild_from_routing_invokes_eager_warmup_after_builds(
    monkeypatch: Any,
) -> None:
    from magos.compression import prebuild_from_routing  # noqa: PLC0415
    from magos.compression.registry import PipelineRegistry  # noqa: PLC0415
    from magos.routing.schema import CompressOptions  # noqa: PLC0415

    cfg = _make_routing_config(rule_compress_options=[CompressOptions()])
    reg = PipelineRegistry()

    seen: list[Any] = []
    monkeypatch.setattr("magos.compression.warmup.eager_warmup", lambda r=None: seen.append(r))

    prebuild_from_routing(cfg, registry=reg)

    assert seen == [reg]
