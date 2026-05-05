# Failure semantics

`compress()` already fails open (`compress.py:311-324`):

- Wraps the pipeline in `try/except`.
- Calls `get_otel_metrics().record_compression_failure(...)`.
- Returns `CompressResult(messages=messages, tokens_*=0)` (original
  messages, zero metrics).

Magos does **not** wrap this. We do swallow import errors as a defence
against optional extras (kompress weights, etc.) being missing: log
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
