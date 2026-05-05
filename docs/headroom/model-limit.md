# `model_limit` resolution

`compress(messages, model, model_limit=...)` accepts an int with default
200000 (`compress.py:161`). Two transforms consume it:

- **`IntelligentContextManager`** (`intelligent_context.py:200-227`):
  the over-budget gate. If `current_tokens > model_limit - output_buffer`,
  message dropping fires.
- **`ContentRouter`** (`content_router.py:1516-1533`): computes
  `context_pressure = tokens_before / model_limit` and linearly
  interpolates between relaxed and aggressive compression thresholds.

`CacheAligner` doesn't use it; prefix stabilisation is model-agnostic.

Headroom's higher-level integrations (`HeadroomClient`, langchain,
agno, strands, ASGI proxy) all auto-detect `model_limit` per request
via per-provider `get_context_limit(model)` (`providers/base.py:69`).
The LiteLLM provider implementation
(`providers/litellm.py:184`) calls `litellm.get_model_info(model)` and
reads `max_input_tokens`.

The simple `headroom.compress()` API skips all of this and trusts the
caller. Magos's `_apply_compress` (`rewrites/compress.py`) resolves
`model_limit` itself via `_resolve_model_limit(dispatch_model,
registry=...)`, walking three sources in order:

1. The model registry, if loaded: picks `context_size` off the
   matching `ModelEntry`. Bypasses the LiteLLM call entirely.
2. `litellm.get_model_info(dispatch_model)`: reads `max_input_tokens`
   (or `max_tokens`). Cached per dispatch id, success and fallback
   both, so unknown models don't keep retriggering LiteLLM's noisy
   "model not mapped" stderr print.
3. Hardcoded `200_000` default.

Operators can override per-rule with `compress.model_limit: <int>`,
which wins over all three.

Failure modes for the lookup:

- Unknown model id -> exception -> fallback to 200000.
- Model id with `@suffix` (Anthropic 1M variant) -> exception ->
  fallback. If using such a model, set `model_limit: 1000000`
  explicitly on the rule.
- LiteLLM not installed -> import error -> fallback.

The fallback default 200000 matches Anthropic's typical context
window. It's wrong for OpenAI models (128K, IntelligentContext won't
fire when it should) and Claude Opus 4.7 (1M, fires too eagerly).
This is why we always do the lookup rather than rely on the default.

## Defaults that matter

`CompressConfig` defaults (`compress.py:77-135`):

| Field                       | Default | Notes                                      |
|-----------------------------|---------|--------------------------------------------|
| `compress_user_messages`    | `False` | user turns skipped (coding-agent profile)  |
| `compress_system_messages`  | `True`  | system content compressed                  |
| `protect_recent`            | `4`     | last N messages untouched                  |
| `protect_analysis_context`  | `True`  | detect "analyze"/"review" intent           |
| `target_ratio`              | `None`  | model decides (~15% kept, aggressive)      |
| `min_tokens_to_compress`    | `250`   | shorter messages pass through              |
| `kompress_model`            | `None`  | `null` -> default `chopratejas/kompress-base`; `"disabled"` -> skip ML compression entirely (only SmartCrusher + CacheAligner run); `"<hf-id>"` -> custom model. See the Kompress section above. |

Operators: with these defaults, the prefix tends to become *more*
cache-stable, not less. The cache-invalidation risk surfaces when
operator overrides default to e.g. `protect_recent=0` +
`compress_system_messages=True` + aggressive `target_ratio` against a
hot cache.
