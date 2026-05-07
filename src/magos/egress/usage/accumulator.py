"""``UsageAccumulator``: stateful streaming usage aggregator.

Walks the ``Shape``'s ``stream_events`` declaration to find the right
event name, the path to the usage dict within event data, and the
per-field key paths inside it. Wire-format-specific facts live on
:class:`magos.shapes.Shape`, not here.
"""

from __future__ import annotations

from typing import Any

from magos.shapes import Shape, Usage
from magos.shapes.base import _safe_int, _walk

_FIELD_ATTR: dict[str, str] = {
    "input": "_input",
    "output": "_output",
    "cache_read": "_cache_read",
    "cache_write": "_cache_write",
}


class UsageAccumulator:
    """Stateful usage accumulator fed parsed SSE events as the stream passes."""

    def __init__(self, shape: Shape) -> None:
        self._shape = shape
        self._input = 0
        self._output = 0
        self._cache_read = 0
        self._cache_write = 0
        self._model: str | None = None

    @property
    def model(self) -> str | None:
        return self._model

    def snapshot(self) -> Usage:
        return Usage(
            input=self._input,
            output=self._output,
            cache_read=self._cache_read,
            cache_write=self._cache_write,
        )

    def feed(self, event_name: str | None, data: dict[str, Any]) -> None:
        for ev in self._shape.stream_events:
            if ev.event_name is not None and ev.event_name != event_name:
                continue
            usage = _walk(data, ev.usage_path)
            if not isinstance(usage, dict):
                continue
            for canonical, key_path in ev.fields.items():
                setattr(self, _FIELD_ATTR[canonical], _safe_int(_walk(usage, key_path)))
            if ev.model_path is not None:
                model = _walk(data, ev.model_path)
                if isinstance(model, str):
                    self._model = model
