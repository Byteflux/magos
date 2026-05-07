"""``UsageAccumulator``: stateful per-shape SSE event aggregator for streaming."""

from __future__ import annotations

from typing import Any

from magos.shapes import Shape

from .core import Usage, _safe_int


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
        if self._shape == "anthropic":
            self._feed_anthropic(event_name, data)
        elif self._shape == "openai-chat":
            self._feed_openai_chat(data)
        else:
            self._feed_openai_responses(event_name, data)

    def _feed_anthropic(self, event_name: str | None, data: dict[str, Any]) -> None:
        # Input + cache arrive on ``message_start.message.usage``; final
        # output arrives on ``message_delta.usage``.
        if event_name == "message_start":
            message = data.get("message")
            if isinstance(message, dict):
                u = message.get("usage")
                if isinstance(u, dict):
                    self._input = _safe_int(u.get("input_tokens"))
                    self._cache_read = _safe_int(u.get("cache_read_input_tokens"))
                    self._cache_write = _safe_int(u.get("cache_creation_input_tokens"))
                model = message.get("model")
                if isinstance(model, str):
                    self._model = model
        elif event_name == "message_delta":
            u = data.get("usage")
            if isinstance(u, dict):
                output = _safe_int(u.get("output_tokens"))
                if output:
                    self._output = output

    def _feed_openai_chat(self, data: dict[str, Any]) -> None:
        # Usage only on the terminal chunk, gated on
        # ``stream_options.include_usage: true``.
        u = data.get("usage")
        if isinstance(u, dict):
            self._input = _safe_int(u.get("prompt_tokens"))
            self._output = _safe_int(u.get("completion_tokens"))
            details = u.get("prompt_tokens_details")
            if isinstance(details, dict):
                self._cache_read = _safe_int(details.get("cached_tokens"))
        model = data.get("model")
        if isinstance(model, str):
            self._model = model

    def _feed_openai_responses(self, event_name: str | None, data: dict[str, Any]) -> None:
        # Usage arrives on ``response.completed.response.usage``.
        if event_name == "response.completed":
            response = data.get("response")
            if isinstance(response, dict):
                u = response.get("usage")
                if isinstance(u, dict):
                    self._input = _safe_int(u.get("input_tokens"))
                    self._output = _safe_int(u.get("output_tokens"))
                    details = u.get("input_tokens_details")
                    if isinstance(details, dict):
                        self._cache_read = _safe_int(details.get("cached_tokens"))
                model = response.get("model")
                if isinstance(model, str):
                    self._model = model
