"""Body translation for non-Anthropic dispatch.

Three transforms run when the dispatch target is not Anthropic:

- `output_config` -> OpenAI `reasoning_effort` / `response_format`
- `additionalProperties: {}` -> `additionalProperties: True` in
  schema-bearing fields (some openai-compatible upstreams reject the
  empty-object form)
- drop unknown Anthropic-only fields that would leak into the
  destination SDK as `unexpected keyword argument`

All three are no-ops for Anthropic-bound traffic so prompt-cache
byte-equivalence is preserved.
"""

from __future__ import annotations

from typing import Any

from magos.telemetry import get_logger

log = get_logger("magos.dispatch.translate")

# Fields LiteLLM's `anthropic_messages` translator maps to non-Anthropic
# providers; anything else leaks via `**kwargs` into the destination SDK.
_ANTHROPIC_MESSAGES_CANONICAL_FIELDS: frozenset[str] = frozenset(
    {
        "model",
        "messages",
        "max_tokens",
        "system",
        "temperature",
        "top_p",
        "top_k",
        "stop_sequences",
        "stream",
        "metadata",
        "tools",
        "tool_choice",
        "thinking",
        # OpenAI-shape extras produced by `_translate_output_config`;
        # ride `**kwargs` to the destination translator.
        "reasoning_effort",
        "response_format",
    }
)

# Anthropic accepts `xhigh`/`max`; OpenAI's `reasoning_effort` tops
# out at `high`. `minimal` is OpenAI-only and never inbound.
_ANTHROPIC_EFFORT_TO_OPENAI: dict[str, str] = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}

# Top-level fields that legitimately carry JSON Schema; `messages`
# never does, so excluding it avoids scanning the bulk of every body.
_SCHEMA_BEARING_FIELDS: tuple[str, ...] = ("tools", "tool_choice", "response_format")


def _translate_output_config(body: dict[str, Any]) -> dict[str, Any]:
    """Map Anthropic `output_config` to OpenAI `reasoning_effort` / `response_format`.

    Caller-supplied `reasoning_effort` / `response_format` win over
    the derived values. `xhigh`/`max` effort clamps to `high`.
    """
    cfg = body.get("output_config")
    if not isinstance(cfg, dict):
        return body
    out = {k: v for k, v in body.items() if k != "output_config"}
    effort = cfg.get("effort")
    if isinstance(effort, str) and "reasoning_effort" not in out:
        mapped = _ANTHROPIC_EFFORT_TO_OPENAI.get(effort)
        if mapped is not None:
            out["reasoning_effort"] = mapped
    fmt = cfg.get("format")
    if isinstance(fmt, dict) and fmt.get("type") == "json_schema" and "response_format" not in out:
        # Anthropic nests the schema directly under `format`; OpenAI wraps
        # it in a `json_schema` object. The schema body itself is identical.
        schema = fmt.get("schema")
        if isinstance(schema, dict):
            out["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": fmt.get("name", "response"),
                    "schema": schema,
                    **({"strict": True} if fmt.get("strict") else {}),
                },
            }
    return out


def _coerce_empty_additional_properties(body: dict[str, Any]) -> dict[str, Any]:
    """Replace `additionalProperties: {}` with `true` in schema-bearing fields.

    Semantically identical per JSON Schema, but some openai-compatible
    upstreams (Vultr) reject the empty-object form. Walks only schema-
    bearing top-level fields and shares storage on unchanged subtrees.
    """
    updates: dict[str, Any] = {}
    for field in _SCHEMA_BEARING_FIELDS:
        if field not in body:
            continue
        new_value = _coerce_empty_ap(body[field])
        if new_value is not body[field]:
            updates[field] = new_value
    if not updates:
        return body
    log.info("anthropic.coerced_empty_additional_properties")
    return {**body, **updates}


def _coerce_empty_ap(value: Any) -> Any:
    """Return `value` with empty-dict `additionalProperties` coerced to True.

    Returns the input by reference if no coercion was needed -- the caller
    uses `is` to detect changes, so unchanged subtrees share storage.
    """
    if isinstance(value, dict):
        new_pairs: dict[str, Any] | None = None
        for key, child in value.items():
            if key == "additionalProperties" and isinstance(child, dict) and not child:
                new_pairs = new_pairs or dict(value)
                new_pairs[key] = True
                continue
            new_child = _coerce_empty_ap(child)
            if new_child is not child:
                new_pairs = new_pairs or dict(value)
                new_pairs[key] = new_child
        return new_pairs if new_pairs is not None else value
    if isinstance(value, list):
        new_items: list[Any] | None = None
        for index, item in enumerate(value):
            new_item = _coerce_empty_ap(item)
            if new_item is not item:
                new_items = new_items or list(value)
                new_items[index] = new_item
        return new_items if new_items is not None else value
    return value


def strip_anthropic_extras(
    body: dict[str, Any], dispatch_model: str, *, client_model: str
) -> dict[str, Any]:
    """Translate `output_config`, coerce empty `additionalProperties`,
    drop unknown Anthropic-only fields. No-op for Anthropic-bound traffic.
    """
    if dispatch_model.startswith("anthropic/"):
        return body
    body = _translate_output_config(body)
    body = _coerce_empty_additional_properties(body)
    extras = set(body) - _ANTHROPIC_MESSAGES_CANONICAL_FIELDS
    if not extras:
        return body
    log.info(
        "anthropic.dropped_unknown_fields",
        model=client_model,
        dispatch_model=dispatch_model,
        fields=sorted(extras),
    )
    return {k: v for k, v in body.items() if k in _ANTHROPIC_MESSAGES_CANONICAL_FIELDS}
