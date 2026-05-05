# Translation

## Anthropic-shape cross-provider translation

`/v1/messages` against a non-Anthropic upstream (e.g. an OpenAI-shaped
provider mapped via routing) goes through
`litellm.anthropic_messages`. LiteLLM accepts Anthropic-shape *in* and
emits Anthropic-shape *out* regardless of upstream provider, but two
preprocessing steps happen in `egress/translate/anthropic.py` first:

- **Anthropic-only fields stripped** for non-Anthropic upstreams
  (`_strip_anthropic_extras`): `context_management` and similar fields
  LiteLLM passes through as `**kwargs` and that the upstream provider
  doesn't understand.
- **`output_config.effort` to `reasoning_effort`** translation. Anthropic
  uses `output_config.effort` (`low|medium|high|xhigh|max`); OpenAI
  uses `reasoning_effort` (`low|medium|high`). Magos clamps
  `xhigh`/`max` → `high`.
- **`additionalProperties: {}` to `additionalProperties: true`** in tool
  `input_schema` blocks (`_coerce_empty_additional_properties`). Anthropic's
  Messages API accepts the empty-object form (an empty schema means
  "any extras allowed, no constraints"); some openai-compatible upstreams
  -- Vultr's `custom_openai`-routed inference, in particular -- run the
  request through a metaschema validator that misreports `{}` as `[]` and
  rejects with `[] is not of type 'object', 'boolean'`. The two forms
  (`{}` and `true`) are semantically identical per the JSON Schema spec,
  so the coercion is safe and sidesteps the upstream bug. Walks the entire
  body so the rule catches schemas wherever they appear -- not just
  `tools[*].input_schema` but also `response_format.json_schema.schema`,
  `tool_choice`, and any nested `properties` / `items` blocks.

If you add a new Anthropic-only field downstream, mirror it in the
strip list. If you discover another shape that Anthropic accepts and an
openai-shape upstream rejects, add the coercion to
`_coerce_empty_additional_properties` (or a sibling) -- always
unidirectional (Anthropic-bound traffic is left verbatim) and only when
the destination semantics are identical.
