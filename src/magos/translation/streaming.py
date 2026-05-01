"""Stateful translator from OpenAI streaming chunks to Anthropic SSE events."""

from __future__ import annotations

import secrets
from typing import Any

from magos.translation._models import AnthropicStopReason
from magos.translation._shared import FINISH_TO_STOP


class AnthropicStreamTranslator:
    """Convert a stream of OpenAI Chat Completions chunks into Anthropic events.

    Anthropic's Messages streaming format is a stateful sequence:
    ``message_start``, then per content block ``content_block_start`` /
    ``content_block_delta`` / ``content_block_stop``, then ``message_delta``
    and ``message_stop``. This class buffers the cross-chunk state needed to
    emit that sequence from OpenAI's flatter chunk stream.

    Tool-call argument JSON is forwarded as ``input_json_delta`` slices; the
    client is responsible for reassembling and parsing.

    ``input_tokens`` is set to 0 in ``message_start`` since OpenAI streaming
    chunks do not carry prompt token counts; ``output_tokens`` is updated from
    a final usage chunk if present (e.g. ``stream_options.include_usage``).
    """

    def __init__(self, *, message_id: str | None = None) -> None:
        self._message_id = message_id or "msg_" + secrets.token_hex(12)
        self._started = False
        self._stopped = False
        self._model = ""
        self._next_block_index = 0
        self._open_block_index: int | None = None
        self._open_block_kind: str | None = None  # "text" | "tool_use"
        self._tool_index_map: dict[int, int] = {}  # OpenAI tool index -> Anthropic block index
        self._stop_reason: AnthropicStopReason | None = None
        self._output_tokens = 0

    def feed(self, chunk: dict[str, Any]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []

        if not self._started:
            self._started = True
            self._model = str(chunk.get("model", ""))
            events.append(
                {
                    "type": "message_start",
                    "message": {
                        "id": self._message_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": self._model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                }
            )

        usage = chunk.get("usage")
        if isinstance(usage, dict):
            ct = usage.get("completion_tokens")
            if isinstance(ct, int):
                self._output_tokens = ct

        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            tool_calls = delta.get("tool_calls")
            finish = choice.get("finish_reason")

            if content:
                events.extend(self._handle_text_delta(content))
            if tool_calls:
                events.extend(self._handle_tool_call_deltas(tool_calls))
            if finish:
                self._stop_reason = FINISH_TO_STOP.get(finish, "end_turn")

        return events

    def finish(self) -> list[dict[str, Any]]:
        if self._stopped:
            return []
        self._stopped = True
        events: list[dict[str, Any]] = self._close_open_block()
        events.append(
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": self._stop_reason or "end_turn",
                    "stop_sequence": None,
                },
                "usage": {"output_tokens": self._output_tokens},
            }
        )
        events.append({"type": "message_stop"})
        return events

    def _handle_text_delta(self, text: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if self._open_block_kind != "text":
            events.extend(self._close_open_block())
            idx = self._next_block_index
            self._next_block_index += 1
            self._open_block_index = idx
            self._open_block_kind = "text"
            events.append(
                {
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {"type": "text", "text": ""},
                }
            )
        events.append(
            {
                "type": "content_block_delta",
                "index": self._open_block_index,
                "delta": {"type": "text_delta", "text": text},
            }
        )
        return events

    def _handle_tool_call_deltas(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for tc in tool_calls:
            oai_idx = int(tc.get("index", 0))
            fn = tc.get("function") or {}

            if oai_idx not in self._tool_index_map:
                events.extend(self._close_open_block())
                idx = self._next_block_index
                self._next_block_index += 1
                self._tool_index_map[oai_idx] = idx
                self._open_block_index = idx
                self._open_block_kind = "tool_use"
                events.append(
                    {
                        "type": "content_block_start",
                        "index": idx,
                        "content_block": {
                            "type": "tool_use",
                            "id": tc.get("id") or "",
                            "name": fn.get("name") or "",
                            "input": {},
                        },
                    }
                )

            args = fn.get("arguments")
            if args:
                events.append(
                    {
                        "type": "content_block_delta",
                        "index": self._tool_index_map[oai_idx],
                        "delta": {"type": "input_json_delta", "partial_json": args},
                    }
                )
        return events

    def _close_open_block(self) -> list[dict[str, Any]]:
        if self._open_block_index is None:
            return []
        ev: list[dict[str, Any]] = [{"type": "content_block_stop", "index": self._open_block_index}]
        self._open_block_index = None
        self._open_block_kind = None
        return ev
