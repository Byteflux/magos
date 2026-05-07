# The compression pipeline

## How magos drives the pipeline

`magos.compression` (peer package to `routing/`, `egress/`, etc.) owns a
per-(config-fingerprint, provider) registry of `TransformPipeline`
instances. The `compress` routing rewrite calls
`magos.compression.apply(...)` rather than Headroom's `compress()`
library entry. This buys us:

- Per-rule transform configuration via `CompressOptions` knobs
  (`smart_routing`, `code_aware`, `intelligent_context`,
  `keep_last_turns`).
- Provider-bound pipelines (Anthropic for `/v1/messages` family, OpenAI
  for `/v1/chat/completions`) so token counting matches the destination.
- Prefix-cache awareness via `frozen_message_count`: the routing rewrite
  fetches a per-session `PrefixCacheTracker` from `magos.cache.get_store()`,
  reads how many leading messages the upstream had already cached on the
  previous turn, and passes that to `pipeline.apply` so the pipeline
  doesn't bust the cache. After the response, a `post_response_hook`
  registered on the routing layer feeds the upstream's reported
  `cache_read` / `cache_write` tokens back into the tracker.
- Token-inflation guard: if the pipeline produces more tokens than it
  received, the wrapper reverts to the original messages.
- Per-rule pipeline pre-warm: at startup, `MagosCompressionWarmup`
  calls `magos.compression.prebuild_from_routing(cfg)`, which walks the
  loaded routing config and pre-builds a pipeline for every distinct
  (token-mode `CompressOptions`, provider) tuple. This eliminates first-
  request cold-start latency for rules that override `smart_routing`,
  `code_aware`, `intelligent_context`, or `keep_last_turns`. Cache-mode
  Compress is skipped (CacheAligner has no transform model loads). After
  the builds, `eager_warmup` walks each unique transform calling
  `eager_load_compressors()`.

`mode: cache` continues to use the standalone `CacheAligner` in
`cache_mode.py`; it does not go through the registry.

The transform list magos builds (`src/magos/compression/build.py`) for
the default `PipelineConfig`:

```
CacheAligner(disabled) -> ContentRouter -> IntelligentContextManager
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
  `sentence_transformers` to `sklearn` to `pandas` to `pyarrow`, and pyarrow's
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
    which does `import litellm` at module top, and litellm pulls in
    PyO3 Rust bindings (cryptography, tokenizers) on import. Once any
    PyO3 ext has initialized on the main thread, importing pyarrow's
    `.pyd` (transitively via sentence_transformers) crashes during
    `create_module`, on **either** the main thread or a worker thread.
    Doing the preload before that import wins the race.
  - `_preload_sentence_transformers()` in `rewrites/compress/_preload.py` runs
    immediately before any headroom import inside `_apply_compress`
    and `_apply_cache_aligner`. Belt-and-suspenders for callers that
    don't go through `cli/serve.py` (e.g. tests, embedded use).
  - `tests/conftest.py` does the preload at session start. Required
    because `tests/ingress/mitm/test_addon.py` imports `mitmproxy.http` which loads
    cryptography Rust before the cache-align test gets a chance to
    preload.

  **Why the request-time preload alone isn't sufficient**: `route()`
  runs on a worker thread, so the request-time preload also runs on a
  worker thread, and pyarrow's `.pyd` load on a worker thread after
  PyO3 has initialized on the main thread crashes the same way (the
  same `create_module` failure, just observed from a different thread).
  The CLI-level preload runs on the main thread before any PyO3 ext is
  loaded, so both subsequent main-thread and worker-thread imports of
  pyarrow find it already cached and skip `create_module` entirely.
  ONNX backend tends to surface this faster because nothing on the
  ONNX kompress code path imports sentence_transformers at startup, so
  the very first import of it is on a worker thread when the first
  compress request lands. The PyTorch backend usually transitively
  imports it earlier as a side effect of `transformers` /
  `safetensors` setup.

  The cascade is **import-time**, not runtime: `dynamic_detector.py`
  unconditionally imports `spacy` and `sentence_transformers` at module
  top level (wrapped in `try/except ImportError` for graceful
  degradation, which doesn't help when the deps *are* installed).
  Default `detection_tiers=['regex']` means none of the ML code
  actually runs: Tier 1 regex covers UUIDs, request IDs, sessions,
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

Operators can swap the model: domain-specific variants like
`chopratejas/kompress-finance` are referenced in the docstring example.
Magos exposes this via `CompressOptions.kompress_model` (three-way
switch: `null` -> default model, `"disabled"` -> skip ML entirely,
`"<hf-model-id>"` -> custom model). All three values map onto Headroom's
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
is **sequential per chunk**: long content (many 350-word chunks)
translates to many forward passes. No batching across chunks.

Two compression modes:

- **No `target_ratio`** (model decides): keep/discard from the binary
  head, span rescues borderline tokens. Compression ratio emerges from
  content.
- **`target_ratio` set** (e.g. `0.3`): score per token, rank globally,
  keep top-N. Forces a specific ratio regardless of content
  compressibility. Aggressive; only set when you're sure.

Kompress weights are cached at module level (`_kompress_cache: dict`,
thread-locked) keyed by `model_id`, so repeated rules with the same
model are cheap. Multiple models can coexist. There's also
`unload_kompress_model()` for memory pressure.

Kompress **is** preloaded by magos at startup. The
`MagosCompressionWarmup` lifespan component first calls
`magos.compression.prebuild_from_routing(cfg)` to build every
(`PipelineConfig`, provider) pipeline implied by token-mode `compress`
rewrites, then `eager_warmup(registry)` walks each unique transform and
invokes `eager_load_compressors()`. `ContentRouter.eager_load_compressors`
loads Kompress (subject to `kompress_model` / `enable_kompress`),
the Magika content detector, and — if `code_aware` is set — tree-sitter
parsers for the common languages. The very first compress request after
startup therefore hits a hot pipeline.

Gotchas:

- **`chunk_words=350` is model-coupled.** Custom models trained on
  different chunk sizes need matching `chunk_words` and
  `score_threshold`; Headroom doesn't validate, mismatched chunk
  sizes silently produce worse compression.
- **`question` parameter is ignored** (`kompress_compressor.py:362`):
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

## CCR (reversible compression)

When the compression pipeline emits a marker like `[N items compressed
to M. Retrieve more: hash=abc123]`, magos can let the model retrieve
the original content on demand:

- The compress rewrite (token mode) scans its post-compression messages
  for markers via `headroom.ccr.CCRToolInjector`. If markers are found,
  it injects the `headroom_retrieve` tool definition into `body.tools`
  and (when no prefix is frozen) prepends a system-message instruction
  block describing how to call the tool.
- Egress dispatch (`magos.egress.dispatch`) wraps the upstream response
  with `magos.ccr.wrap_response` (non-streaming) or `magos.ccr.wrap_stream`
  (streaming). The wrappers short-circuit when the request didn't carry
  the CCR tool. When the model calls `headroom_retrieve`, the wrappers
  retrieve from `headroom.cache.compression_store`, build a continuation
  closure that re-runs the translate adapter with the original retrieval
  results appended, and return the final response to the client.
- CCR is on by default whenever compress is on. Disable per-rule with
  `compress.ccr_enabled: false`. For finer control, the
  `ccr_inject_tool` and `ccr_inject_instructions` flags scope the
  injection separately.
- v1 supports `target.gateway: translate` only. Passthrough-mode CCR is
  deferred. `/v1/responses` does not support compress in token mode,
  so CCR is naturally out for that endpoint.
