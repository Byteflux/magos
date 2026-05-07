# Failure semantics

Magos owns failure handling at two layers:

- **`magos.compression.apply` inflation guard** (`pipeline.py:72-86`):
  if the pipeline returns `tokens_after > tokens_before`, log
  `compress.inflation_reverted` and swap the result back to the
  original messages with `tokens_saved=0` and `inflation_reverted=True`.
- **`magos.compression.eager_warmup` per-transform try/except**
  (`warmup.py`): a single transform failing to load doesn't break
  process startup; logs `compress.eager_load_failed` and continues.

Headroom's own `compress()` library entry also fails open (returns
original messages with zeroed metrics on exception), but magos calls
`pipeline.apply(...)` directly and does not rely on that guarantee.

## Pipeline init cost

Two distinct init costs:

1. **TransformPipeline construction.** `magos.compression.PipelineRegistry`
   is a thread-locked lazy cache keyed by `(config-fingerprint,
   provider_name)`. First `get_or_build` for a key constructs the
   pipeline, the underlying tokenizer, and the transform instances;
   subsequent calls reuse. Magos warms the default `PipelineConfig` for
   both providers once at FastAPI startup via `MagosCompressionWarmup`
   in `api/lifespan/components.py`, but only when at least one routing
   rule uses `compress`.

2. **Kompress weight load.** Separate from pipeline construction
   itself but warmed at the same lifespan step. Magos's
   `MagosCompressionWarmup` component calls
   `magos.compression.prebuild_from_routing(cfg)`, which builds every
   `(PipelineConfig, provider)` pipeline implied by token-mode
   `compress` rewrites and then `eager_warmup`s each unique transform.
   `eager_warmup` invokes `eager_load_compressors()` on every
   transform that exposes one — `ContentRouter.eager_load_compressors`
   loads Kompress, the Magika content detector, and (when `code_aware`
   is set) tree-sitter parsers. Cold start: tens of seconds on a
   fresh deployment paid up-front at startup, hundreds of ms on a
   restart with cache. Operators who want zero ML cost can declare
   `kompress_model: disabled` per-rule, which bypasses Kompress
   entirely while keeping the rest of the pipeline.

Operators who want zero per-request ML cost can declare
`kompress_model: disabled` per-rule; that bypasses Kompress entirely
while keeping the rest of the pipeline (CacheAligner, SmartCrusher,
non-ML compressors).
