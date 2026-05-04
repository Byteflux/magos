"""SSE frame helpers used by both OpenAI translate paths.

Anthropic streams pass through bytes verbatim and don't use these.
"""

from __future__ import annotations

import json
from typing import Any


def sse_event(data: str) -> bytes:
    return f"data: {data}\n\n".encode()


def sse_named_event(event: dict[str, Any]) -> bytes:
    """OpenAI Responses streaming uses ``event:`` + ``data:`` lines per chunk."""
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
