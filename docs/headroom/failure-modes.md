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
   in `ingress/http/lifespan.py`, but only when at least one routing
   rule uses `compress`.

2. **Kompress weight load.** Separate from pipeline construction.
   `KompressCompressor._get_kompress` is lazy *inside* the compressor:
   first plain-text compress request triggers HF download (or disk
   cache deserialization on subsequent restarts). Magos does not warm
   this. Cold start: tens of seconds on a fresh deployment, hundreds
   of ms on a restart with cache. ContentRouter has a warmup-style
   method (`content_router.py:1232-1239`) that magos could call from
   the lifespan hook to amortise this; not implemented because
   cold-start latency hasn't been a complaint.

Operators who want zero per-request ML cost can declare
`kompress_model: disabled` per-rule; that bypasses Kompress entirely
while keeping the rest of the pipeline (CacheAligner, SmartCrusher,
non-ML compressors).
