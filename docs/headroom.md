# Headroom integration notes

Reference for how Headroom works under magos and the non-obvious findings
behind the integration in `src/magos/routing/rewrites/compress.py`.
Verified against `headroom-ai==0.10.16`.

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

Default ordering (defined upstream in `headroom.compress`):

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
  `sentence_transformers` first. Three preload sites:

  - `_preload_native_load_order()` in `cli/serve.py` runs **before**
    `magos.serve` is imported. This is the load-bearing one: importing
    `magos.serve` transitively pulls in `magos.ingress.http.handlers`,
    which does `import litellm` at module top — and litellm pulls in
    PyO3 Rust bindings (cryptography, tokenizers) on import. Once any
    PyO3 ext has initialized on the main thread, importing pyarrow's
    `.pyd` (transitively via sentence_transformers) crashes during
    `create_module`, on **either** the main thread or a worker thread.
    Doing the preload before that import wins the race.
  - `_preload_sentence_transformers()` in `rewrites/compress.py` runs
    immediately before any headroom import inside `_apply_compress`
    and `_apply_cache_aligner`. Belt-and-suspenders for callers that
    don't go through `cli/serve.py` (e.g. tests, embedded use).
  - `tests/conftest.py` does the preload at session start. Required
    because `tests/ingress/mitm/test_addon.py` imports `mitmproxy.http` which loads
    cryptography Rust before the cache-align test gets a chance to
    preload.

  **Why the request-time preload alone isn't sufficient** since the
  `route()`-on-worker-thread refactor: the request-time preload now
  runs on a worker thread, and pyarrow's `.pyd` load on a worker
  thread after PyO3 has initialized on the main thread also crashes
  (same `create_module` failure, just observed from a different
  thread). The CLI-level preload runs on the main thread before
  any PyO3 ext is loaded, so both subsequent main-thread and
  worker-thread imports of pyarrow find it already cached and skip
  `create_module` entirely. ONNX backend tends to surface this faster
  because nothing on the ONNX kompress code path imports
  sentence_transformers at startup, so the very first import of it
  is on a worker thread when the first compress request lands. The
  PyTorch backend usually transitively imports it earlier as a side
  effect of `transformers` / `safetensors` setup.

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

**`ContentRouter`** (`transforms/content_router.py`) sniffs each
compressible block's content type and dispatches to a per-type
compressor:

| Content type        | Compressor          | Notes                                   |
|---------------------|---------------------|-----------------------------------------|
| `SOURCE_CODE`       | CodeCompressor      | AST-preserving                          |
| `JSON_ARRAY`        | SmartCrusher        | Array dedup                             |
| `SEARCH_RESULTS`    | SearchCompressor    | grep/ripgrep output                     |
| `BUILD_OUTPUT`      | LogCompressor       | Build/test logs                         |
| `GIT_DIFF`          | DiffCompressor      |                                         |
| `HTML`              | HTMLExtractor       | Standard HTML; custom XML tags protected separately |
| `PLAIN_TEXT`        | KompressCompressor  | ML-based; passthrough if backends missing |

Custom XML-style tags (`<system-reminder>`, `<tool_call>`,
`<thinking>`) are protected before ML compression and restored after
(`content_router.py:1073-1112`). Standard HTML tags are routed to
HTMLExtractor instead.

When ContentRouter can't classify a block, the fallback strategy is
also Kompress.

**Kompress in detail.** ModernBERT-based token-level compressor
(`transforms/kompress_compressor.py`). Default model
`chopratejas/kompress-base` on HF, trained on 330K structured tool
outputs per the docstring. Two prediction heads:

- Token head: binary keep/discard classifier per token
  (`kompress_compressor.py:99-122`).
- Span head: 1D CNN producing span-importance scores
  (`:124-130`). Borderline tokens get rescued if their surrounding
  span scores high.

Operators can swap the model — domain-specific variants like
`chopratejas/kompress-finance` are referenced in the docstring example.
Magos exposes this via `CompressOptions.kompress_model` (three-way
switch: `null` → default model, `"disabled"` → skip ML entirely,
`"<hf-model-id>"` → custom model). All three values map onto Headroom's
runtime `kompress_model` kwarg dispatch in
`content_router.py:1299-1339`.

Two backends, ONNX preferred (`kompress_compressor.py:232-253`):

| Backend            | Wheel size | Path                          | Quantization |
|--------------------|-----------|-------------------------------|--------------|
| `onnxruntime`      | ~50MB     | `onnx/kompress-int8.onnx`     | INT8         |
| `torch` + `safetensors` | ~800MB | `model.safetensors` + ModernBERT-base | FP32 |

Our venv has both (`headroom-ai[all]==0.10.16` ships
`onnxruntime==1.25.1` + `torch==2.11.0`). The ONNX path runs.

Per-call mechanics: word-tokenize input, skip if `< 10` words
(passthrough), chunk into windows of `chunk_words=350` (model-coupled
default), per-chunk ModernBERT forward pass, reduce per-token
decisions to per-word keep set, reassemble surviving words. Inference
is **sequential per chunk** — long content (many 350-word chunks)
translates to many forward passes. No batching across chunks.

Two compression modes:

- **No `target_ratio`** (model decides): keep/discard from the binary
  head, span rescues borderline tokens. Compression ratio emerges from
  content.
- **`target_ratio` set** (e.g. `0.3`): score per token, rank globally,
  keep top-N. Forces a specific ratio regardless of content
  compressibility. Aggressive — only set when you're sure.

Kompress weights are cached at module level (`_kompress_cache: dict`,
thread-locked) keyed by `model_id`, so repeated rules with the same
model are cheap. Multiple models can coexist. There's also
`unload_kompress_model()` for memory pressure.

Kompress is **not preloaded by magos**. The lifespan hook calls
`_get_pipeline()` which builds the TransformPipeline but
`KompressCompressor.compress` is lazy (`_get_kompress` only fires when
`_route_and_compress_block` actually hits a PLAIN_TEXT block). First
text-bearing compress request pays the HF download (~tens of seconds
on a fresh deployment) or disk-cache deserialization (~hundreds of
ms). ContentRouter has a warmup-style method that eagerly loads
Kompress (`content_router.py:1232-1239`) — magos doesn't call it.
Worth wiring into the lifespan hook only if cold-start latency
becomes a complaint.

Gotchas:

- **`chunk_words=350` is model-coupled.** Custom models trained on
  different chunk sizes need matching `chunk_words` and
  `score_threshold` — Headroom doesn't validate, mismatched chunk
  sizes silently produce worse compression.
- **`question` parameter is ignored** (`kompress_compressor.py:362`) —
  reserved for future QA-aware compression. Setting it has no effect
  today.
- **ONNX availability check is import-only.** `_is_onnx_available`
  succeeds if `onnxruntime` and `transformers` import. It does not
  validate that the ONNX session can actually load on this machine.

**`IntelligentContextManager`** (`transforms/intelligent_context.py`)

- Conditional, **only fires when over budget**:
  `current_tokens > model_limit - output_buffer`.
- Strategies: `COMPRESS_FIRST` -> `SUMMARIZE` (needs callback) ->
  `DROP_BY_SCORE`.
- Safety guarantees: system messages preserved, last N turns protected,
  tool call/response pairs kept atomic.

## `model_limit` resolution

`compress(messages, model, model_limit=...)` accepts an int with default
200000 (`compress.py:161`). Two transforms consume it:

- **`IntelligentContextManager`** (`intelligent_context.py:200-227`):
  the over-budget gate. If `current_tokens > model_limit - output_buffer`,
  message dropping fires.
- **`ContentRouter`** (`content_router.py:1516-1533`): computes
  `context_pressure = tokens_before / model_limit` and linearly
  interpolates between relaxed and aggressive compression thresholds.

`CacheAligner` doesn't use it — prefix stabilisation is model-agnostic.

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

1. The model registry, if loaded — picks `context_size` off the
   matching `ModelEntry`. Bypasses the LiteLLM call entirely.
2. `litellm.get_model_info(dispatch_model)` — reads `max_input_tokens`
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
window. It's wrong for OpenAI models (128K — IntelligentContext won't
fire when it should) and Claude Opus 4.7 (1M — fires too eagerly).
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

Two distinct init costs:

1. **TransformPipeline construction.** `_get_pipeline()` is a
   thread-locked lazy singleton in upstream `headroom.compress` (magos
   imports it via `ingress/http/lifespan.py:129`). First call
   constructs the pipeline, the underlying tokenizer, transform
   instances. Subsequent calls reuse. Magos warms this once at FastAPI
   startup via the lifespan hook in `ingress/http/lifespan.py` if any routing rule
   uses `compress`.

2. **Kompress weight load.** Separate from pipeline construction.
   `KompressCompressor._get_kompress` is lazy *inside* the compressor —
   first plain-text compress request triggers HF download (or disk
   cache deserialization on subsequent restarts). Magos does not warm
   this. Cold start: tens of seconds on a fresh deployment, hundreds
   of ms on a restart with cache. ContentRouter has a warmup-style
   method (`content_router.py:1232-1239`) that magos could call from
   the lifespan hook to amortise this; not implemented because
   cold-start latency hasn't been a complaint.

Operators who want zero per-request ML cost can declare
`kompress_model: disabled` per-rule — that bypasses Kompress entirely
while keeping the rest of the pipeline (CacheAligner, SmartCrusher,
non-ML compressors).

## Forcing the Kompress backend

`MagosSettings.kompress_backend` (env: `MAGOS_KOMPRESS_BACKEND`)
controls which Kompress backend Headroom uses:

| Value         | Behaviour                                                                 |
|---------------|---------------------------------------------------------------------------|
| `auto` (default) | Headroom prefers ONNX Runtime when installed, falls back to PyTorch. INT8 ONNX runs CPU-only out of the box (Headroom hardcodes `providers=["CPUExecutionProvider"]`). |
| `pytorch`     | Forces the PyTorch backend. `_load_kompress_pytorch` auto-selects CUDA / MPS / CPU via `device='auto'`. This is the path to choose for GPU acceleration. |

Implementation: when set to `pytorch`, the FastAPI lifespan hook
replaces `headroom.transforms.kompress_compressor._is_onnx_available`
with a False-returning stub. Headroom's `_load_kompress` resolves that
name from the module namespace at call time, so the override flips
backend selection without patching Headroom itself. See
`_force_kompress_pytorch` in `ingress/http/lifespan.py`.

Caveats:

- The override is process-wide. Per-rule backend selection isn't
  available because Kompress weights are cached at module level keyed
  by `model_id`, not by backend.
- `pytorch` requires `torch` (and `safetensors` + `transformers`) to
  be installed. If they're missing, the first compress request raises
  `ImportError` from Headroom — magos's lazy import catch logs
  `compress.import_failed` and the rule no-ops.
- For GPU, you also need a CUDA-enabled `torch` build. The default
  PyPI `torch` wheels include CPU + CUDA on Linux/Windows; macOS
  builds ship MPS for Apple Silicon. Check `torch.cuda.is_available()`
  to verify GPU availability.
- The override fires unconditionally at lifespan startup when
  `kompress_backend=pytorch`, regardless of whether any rule actually
  uses `compress`. The cost is one attribute assignment — no I/O, no
  model load.

Why we don't expose ONNX CUDA via this knob: even with `onnxruntime-gpu`
installed, Headroom's `_load_kompress_onnx` hardcodes
`providers=["CPUExecutionProvider"]` (`kompress_compressor.py:179-183`),
so flipping `_is_onnx_available` doesn't help — the ONNX session would
still be CPU-bound. A working ONNX-CUDA path needs an upstream Headroom
patch to thread an EP list through, plus careful handling of INT8
operator coverage on CUDA EP (many INT8 ops fall back to CPU). See
prior research notes; not pursued.

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
| `/v1/responses`                   | `instructions` | **`mode: cache` only** |
| `/v1/responses`                   | `input`        | unsupported — different shape from `messages`, no upstream Headroom path; `mode: token` silently no-ops |
| `/v1/responses/{id}` and friends  | n/a            | no-op (no body to compress)                         |

The Responses `instructions` string is wrapped as a synthetic
`[{"role": "system", "content": instructions}]` and fed to CacheAligner.
The aligner mutates the message's `content` in place; we read it back
and write it to `instructions`. No new messages are introduced. See
`_apply_compress_responses` in `rewrites/compress.py`.

Compressing `input` is not implemented. It would require a round-trip
converter for `input_text` / `message` / `function_call` / etc. items,
including atomicity preservation for `function_call` ↔
`function_call_output` pairs. Headroom's `HeadroomCallback`
(`integrations/litellm_callback.py`) explicitly filters
`call_type ∉ {completion, acompletion}` and only reads `data["messages"]`,
so there's no upstream conversion to mirror.

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

| File                                            | Purpose                                              |
|--------------------------------------------------|------------------------------------------------------|
| `src/magos/routing/schema.py`                    | `Compress`, `CompressOptions`, `CompressMode` schema |
| `src/magos/routing/rewrites/compress.py`         | `_apply_compress`, `_apply_cache_aligner`, model_limit resolution |
| `src/magos/routing/loader.py`                    | `Compress` listed in `_rewrites_touch_body`          |
| `src/magos/ingress/http/lifespan.py`             | Lifespan warmup hook + kompress backend override     |
| `tests/routing/rewrites/test_compress.py`        | Unit tests for both modes + endpoint scoping         |
| `tests/routing/rewrites/test_compress_registry.py` | `model_limit` resolution against registry           |
| `tests/routing/test_loader.py`                   | YAML round-trip + body-touch warning                 |
| `tests/ingress/http/test_lifespan.py`            | Lifespan warmup + kompress backend behaviour         |
| `docs/routing.md`                                | Operator-facing rewrite-op docs                      |
