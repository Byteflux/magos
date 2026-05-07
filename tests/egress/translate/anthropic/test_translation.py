"""Body translation: ``output_config`` mapping, ``additionalProperties`` coercion,
unknown-field stripping. Anthropic-bound traffic must be left alone so prompt-
cache byte-equivalence is preserved.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ._helpers import proxy_anthropic_messages


@pytest.mark.unit
def test_proxy_anthropic_messages_strips_unknown_fields_for_non_anthropic() -> None:
    """Anthropic-only body fields must not leak through to non-Anthropic upstreams.

    LiteLLM's ``anthropic_messages`` only translates canonical Anthropic
    Messages fields to OpenAI shape; unknown fields fall through to ``**kwargs``
    and surface inside the destination SDK as ``unexpected keyword argument``
    errors (e.g. ``output_config``, ``context_management`` from Claude Code).
    Verify the proxy strips them before dispatch when the target is not
    Anthropic, and leaves them alone when it is.
    """
    seen: dict[str, Any] = {}

    async def fake(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"id": "ok", "content": []}

    body = {
        "model": "claude-x",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
        "context_management": {"edits": []},
        "output_config": {
            "effort": "xhigh",
            "format": {
                "type": "json_schema",
                "schema": {"type": "object", "properties": {"name": {"type": "string"}}},
            },
        },
    }
    asyncio.run(
        proxy_anthropic_messages(
            body,
            dispatch_model="custom_openai/zai-org/GLM-5.1-FP8",
            completion=fake,
        )
    )
    # Anthropic-only fields stripped; output_config translated to OpenAI shape.
    assert "context_management" not in seen
    assert "output_config" not in seen
    assert seen["messages"] == body["messages"]
    assert seen["reasoning_effort"] == "high"  # xhigh clamps to high
    assert seen["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "schema": {"type": "object", "properties": {"name": {"type": "string"}}},
        },
    }

    seen.clear()
    asyncio.run(
        proxy_anthropic_messages(
            body,
            dispatch_model="anthropic/claude-x",
            completion=fake,
        )
    )
    # Anthropic-bound: forwarded verbatim, no translation.
    assert seen.get("context_management") == {"edits": []}
    assert seen.get("output_config") == body["output_config"]
    assert "reasoning_effort" not in seen
    assert "response_format" not in seen


def test_proxy_anthropic_messages_coerces_empty_additional_properties() -> None:
    """``additionalProperties: {}`` becomes ``true`` for non-Anthropic dispatch.

    Reproduces the Vultr / custom_openai failure: Anthropic accepts
    ``additionalProperties: {}`` (empty schema = "any extras allowed");
    Vultr's metaschema validator misreports it as ``[]`` and rejects with
    ``[] is not of type 'object', 'boolean'``. ``true`` is the same
    semantics and routes cleanly through the validator.

    Anthropic-bound traffic is left alone -- the original ``{}`` flows
    verbatim so prompt-cache byte-equivalence is preserved.
    """
    seen: dict[str, Any] = {}

    async def fake(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"id": "ok", "content": []}

    body = {
        "model": "claude-x",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
        "tools": [
            {
                "name": "tool_with_empty_extras",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "nested": {"type": "object", "additionalProperties": {}},
                    },
                    "additionalProperties": {},
                },
            },
            {
                "name": "tool_with_meaningful_extras",
                "input_schema": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
            },
        ],
    }
    asyncio.run(
        proxy_anthropic_messages(body, dispatch_model="custom_openai/Qwen/X", completion=fake)
    )
    sent = seen["tools"]
    assert sent[0]["input_schema"]["additionalProperties"] is True
    assert sent[0]["input_schema"]["properties"]["nested"]["additionalProperties"] is True
    # Non-empty schema is preserved untouched.
    assert sent[1]["input_schema"]["additionalProperties"] == {"type": "string"}

    # Anthropic-bound: empty {} flows verbatim, not coerced.
    seen.clear()
    asyncio.run(
        proxy_anthropic_messages(body, dispatch_model="anthropic/claude-x", completion=fake)
    )
    assert seen["tools"][0]["input_schema"]["additionalProperties"] == {}
