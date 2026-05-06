"""Detect whether a ``RoutedRequest`` carries the CCR retrieval tool."""

from __future__ import annotations

from headroom.ccr import CCR_TOOL_NAME

from magos.routing.request import RoutedRequest


def is_ccr_request(req: RoutedRequest) -> bool:
    """True when ``req.body['tools']`` contains the ``headroom_retrieve`` tool.

    Recognises both Anthropic shape (top-level ``name``) and OpenAI shape
    (``function.name``). Returns False for missing / empty / malformed tools.
    The compress rewrite is the only place that injects this tool, so
    presence is a self-describing signal that CCR is active for this
    request — no per-request side channel needed.
    """
    tools = req.body.get("tools")
    if not isinstance(tools, list):
        return False
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        # Anthropic shape
        if tool.get("name") == CCR_TOOL_NAME:
            return True
        # OpenAI shape
        function = tool.get("function")
        if isinstance(function, dict) and function.get("name") == CCR_TOOL_NAME:
            return True
    return False
