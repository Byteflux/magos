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
