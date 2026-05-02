# Headroom integration notes

Reference for how Headroom works under magos and the non-obvious findings
behind the integration in `src/magos/routing/rewrites.py`. Verified
against `headroom-ai==0.10.16`.

## What Headroom is

Two distinct subsystems shipped in one package:

1. **Compression pipeline** (`headroom.compress`) — token-reduction
   transforms on `messages` lists.
2. **Cache optimizers** (`headroom.cache.*`) — provider-specific cache
   helpers (Anthropic ephemeral breakpoints, OpenAI prefix caching,
   Google). Insert `cache_control` markers, track prefix hashes, score
   breakpoint placement.

Magos uses (1) directly. (2) is opt-in via `mode: cache` in the
`compress` rewrite, which runs only the `CacheAligner` transform — no
breakpoint insertion. Auto-`cache_control` injection via
`AnthropicCacheOptimizer` is not wired; revisit if cache hit rates need
improving beyond prefix stabilisation.

## Anthropic prompt caching, accurately

- Explicit, breakpoint-based, prefix-matched. Operator inserts
  `{"cache_control": {"type": "ephemeral"}}` blocks.
- Max 4 breakpoints per request, 1024-token minimum, 5-minute TTL
  (extended on hit).
- Cost: 25% write premium, 90% read discount.
- Cache key = prefix content up to each breakpoint, hashed.
- Content **after** the last breakpoint is recomputed every request by
  design.

Implication: "modifying messages breaks the cache" is wrong as a
blanket claim. Modifications before/within a breakpoint invalidate that
breakpoint. Modifications after it are normal operation.

Headroom encodes the constants in `headroom/cache/anthropic.py`:
`ANTHROPIC_MIN_CACHEABLE_TOKENS`, `ANTHROPIC_MAX_BREAKPOINTS`,
`ANTHROPIC_CACHE_TTL_SECONDS`, write/read multipliers.

## The compression pipeline

Default ordering (`compress.py:340-345`):

```
CacheAligner -> ContentRouter -> IntelligentContext
```

**`CacheAligner`** (`transforms/cache_aligner.py`)

- Extracts dynamic content from system prompts (dates, UUIDs, JWTs,
  request IDs, hex hashes, high-entropy strings) and reinserts after the
  static block under a `[Dynamic Context]` separator.
- Normalises whitespace.
- Computes a stable prefix hash; tracks across requests.
- Skips `frozen_message_count` messages (already in provider cache).
- Default `enabled=False` for the standalone transform; the full
  pipeline flips it on. Magos flips it on for `mode: cache`.
- Default `use_dynamic_detector=True` cascades into
  `sentence_transformers -> sklearn -> pandas -> pyarrow`, and pyarrow's
  native `.pyd` segfaults during `create_module` on Windows when
  `cryptography.hazmat.bindings._rust` has already been imported in the
  process. Magos always loads `cryptography` transitively (via
  `mitmproxy.http`) so this fires whenever the cache aligner runs.

  **Minimal repro:**

  ```python
  import cryptography.hazmat.bindings._rust
  import sentence_transformers  # boom
  ```

  Confirmed bisection:

  - `cryptography` top-level alone -> OK (doesn't load the Rust ext).
  - `cryptography.hazmat.bindings._rust` -> `pyarrow` direct -> OK.
  - `cryptography.hazmat.bindings._rust` -> `sentence_transformers` ->
    crashes in pyarrow's `create_module`.
  - Reverse order (`sentence_transformers` first, then cryptography) ->
    OK.

  The crash is order-sensitive and only fires when sentence_transformers
  triggers pyarrow's load *after* the PyO3 Rust runtime has initialised.
  Almost certainly a Windows DLL-init / C-runtime interaction between
  PyO3's TLS state and pyarrow's bundled Arrow C++ runtime. Not
  reproduced on Linux/CI; assumed Windows-only.

  Magos uses Headroom's intended default (`use_dynamic_detector=True`)
  and works around the load-order bug by force-importing
  `sentence_transformers` first. Two preload sites:

  - `_preload_sentence_transformers()` in `rewrites.py` runs
    immediately before any headroom import inside `_apply_compress`
    and `_apply_cache_aligner`. In a pure FastAPI deployment this is
    sufficient because `magos.server` does not transitively load
    cryptography at import time (verified against `litellm` and
    `magos.server`'s `sys.modules` graph).
  - `tests/conftest.py` does the preload at session start. Required
    because `tests/test_addon.py` imports `mitmproxy.http` which loads
    cryptography Rust before the cache-align test gets a chance to
    preload.

  Operators running magos behind the mitmproxy addon process should
  note: that's a *separate* `mitmdump` process, not the magos FastAPI
  process. The two don't share state, so the preload doesn't need to
  cross processes.

  The cascade is **import-time**, not runtime: `dynamic_detector.py`
  unconditionally imports `spacy` and `sentence_transformers` at module
  top level (wrapped in `try/except ImportError` for graceful
  degradation, which doesn't help when the deps *are* installed).
  Default `detection_tiers=['regex']` means none of the ML code
  actually runs — Tier 1 regex covers UUIDs, request IDs, sessions,
  ISO 8601 datetimes, versions, high-entropy identifiers. We pay the
  import to get those, then run regex only at execution time.

  Fallback escape hatch: `use_dynamic_detector=False` switches `apply`
  and `should_apply` to `_compiled_patterns` (legacy `date_patterns`),
  covering only `Current date:`, `Today is`, ISO 8601 datetime,
  `Today's date:`. Use if a deployment can't run the preload (e.g. an
  embedded library that imports cryptography unconditionally before
  any opportunity to inject a preload).

**`ContentRouter`** routes per content type: SmartCrusher (JSON dedup),
CodeCompressor, Kompress (text). No magos-side concerns.

**`IntelligentContextManager`** (`transforms/intelligent_context.py`)

- Conditional, **only fires when over budget**:
  `current_tokens > model_limit - output_buffer`.
- Strategies: `COMPRESS_FIRST` -> `SUMMARIZE` (needs callback) ->
  `DROP_BY_SCORE`.
- Safety guarantees: system messages preserved, last N turns protected,
  tool call/response pairs kept atomic.

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
| `kompress_model`            | `None`  | HF model id, or `"disabled"`               |

Operators: with these defaults, the prefix tends to become *more*
cache-stable, not less. The cache-invalidation risk surfaces when
operator overrides default to e.g. `protect_recent=0` +
`compress_system_messages=True` + aggressive `target_ratio` against a
hot cache.

## Failure semantics

`compress()` already fails open (`compress.py:311-324`):

- Wraps the pipeline in `try/except`.
- Calls `get_otel_metrics().record_compression_failure(...)`.
- Returns `CompressResult(messages=messages, tokens_*=0)` (original
  messages, zero metrics).

Magos does **not** wrap this. We do swallow import errors as a defence
against optional extras (kompress weights, etc.) being missing — log
`compress.import_failed` and pass through.

## Pipeline init cost

`_get_pipeline()` is a thread-locked lazy singleton
(`compress.py:327-347`). First call constructs `TransformPipeline` —
tokenizer init, model loads. Subsequent calls reuse.

Magos warms this once at FastAPI startup via the lifespan hook in
`server.py` if any routing rule uses `compress`. Avoids burying multi-
second latency in the first user request.

## Integration shapes considered

| Shape                                                | Verdict |
|------------------------------------------------------|---------|
| `headroom.compress()` direct call                    | **Adopted.** Wrapped as `Compress` rewrite primitive. |
| `headroom.integrations.litellm_callback.HeadroomCallback` | **Rejected.** Implements LiteLLM's `CustomLogger.async_pre_call_hook`, which only fires when LiteLLM runs as a *proxy server*. Magos uses the LiteLLM SDK (`litellm.acompletion`, `litellm.anthropic_messages`, `litellm.aresponses`). The hook never fires in our architecture — verified by grep: `async_pre_call_hook` exists only under `litellm/proxy/`. |
| `headroom.proxy.handlers.*` (HeadroomProxy)          | **Rejected.** ~6,300 LOC of FastAPI handlers that re-implement provider routing, header forwarding, streaming. Stacking it under magos's mitmproxy + FastAPI duplicates routing. |

## Endpoint scope

| Endpoint                          | Field          | Compress support |
|-----------------------------------|----------------|------------------|
| `/v1/messages`                    | `messages`     | both modes       |
| `/v1/chat/completions`            | `messages`     | both modes       |
| `/v1/messages/count_tokens`       | `messages`     | both modes (useful: post-compression token preview) |
| `/v1/responses`                   | `instructions` | **`mode: cache` only** — Phase 1 |
| `/v1/responses`                   | `input`        | unsupported — different shape from `messages`, no upstream Headroom path; `mode: token` silently no-ops |
| `/v1/responses/{id}` and friends  | n/a            | no-op (no body to compress)                         |

**Phase 1 (shipped):** the Responses `instructions` string is wrapped
as a synthetic `[{"role": "system", "content": instructions}]` and
fed to CacheAligner. The aligner mutates the message's `content` in
place; we read it back and write it to `instructions`. No new messages
are introduced. See `_apply_compress_responses` in `rewrites.py`.

**Phase 2+ (not planned):** compressing `input` would require a
round-trip converter for `input_text`/`message`/`function_call`/etc.
items, including atomicity preservation for `function_call` ↔
`function_call_output` pairs. Headroom's `HeadroomCallback`
(`integrations/litellm_callback.py`) explicitly filters
`call_type ∉ {completion, acompletion}` and only reads `data["messages"]`
— so there's no upstream conversion to mirror. Revisit if operator
demand materialises.

## Magos-Headroom mode terminology

Headroom uses its own "proxy mode" terminology in
`headroom/proxy/modes.py`:

- `PROXY_MODE_TOKEN` — prioritise compression (history may be
  rewritten).
- `PROXY_MODE_CACHE` — prioritise cache stability (freeze prior turns).

Magos mirrors these as the `mode: token | cache` switch on the
`compress` rewrite. The semantics differ in scope:

- Magos `mode: cache` runs **only** `CacheAligner` (no message-level
  changes).
- Magos `mode: token` runs the **full** pipeline (cache-aligned +
  routed + dropped if over budget).

## Subtleties worth not forgetting

- Headroom protects images from cache invalidation in
  `proxy/handlers/anthropic.py:_compress_latest_user_turn_images_cache_safe`
  — only the latest non-frozen user turn's images are touched, leaving
  historical image bytes alone because they're likely cached.
  Magos doesn't (yet) replicate this; if we add image-aware compression
  later, mirror this approach.
- Tool ordering matters for cache stability:
  `AnthropicHandlerMixin._sort_tools_deterministically` exists for this
  reason. We don't reorder tools; magos forwards them verbatim.
- Headroom emits OTel metrics out of the box
  (`headroom.observability.get_otel_metrics`). If we ever wire OTel
  collection in magos, these flow for free.

## Where this lives in magos

| File                                  | Purpose                                              |
|---------------------------------------|------------------------------------------------------|
| `src/magos/routing/models.py`         | `Compress`, `CompressOptions`, `CompressMode` schema |
| `src/magos/routing/rewrites.py`       | `_apply_compress`, `_apply_cache_aligner`            |
| `src/magos/routing/loader.py`         | `Compress` listed in `_rewrites_touch_body`          |
| `src/magos/server.py`                 | Lifespan warmup hook                                 |
| `tests/test_routing_rewrites.py`      | Unit tests for both modes + endpoint scoping         |
| `tests/test_routing_loader.py`        | YAML round-trip + body-touch warning                 |
| `tests/test_server.py`                | Lifespan warmup behaviour                            |
| `docs/routing.md`                     | Operator-facing rewrite-op docs                      |
