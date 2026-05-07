# Subtleties worth not forgetting

- Headroom protects images from cache invalidation in
  `proxy/handlers/anthropic.py:_compress_latest_user_turn_images_cache_safe`:
  only the latest non-frozen user turn's images are touched, leaving
  historical image bytes alone because they're likely cached.
  Magos doesn't (yet) replicate this; if we add image-aware compression
  later, mirror this approach.
- Tool ordering matters for cache stability:
  `AnthropicHandlerMixin._sort_tools_deterministically` exists for this
  reason. We don't reorder tools; magos forwards them verbatim.
- Headroom emits OTel metrics out of the box
  (`headroom.observability.get_otel_metrics`). Magos already ships an
  OTel pipeline (`MAGOS_OTEL_ENABLED=1`, see `docs/cli.md`); these
  meters flow into the same exporter without extra wiring.

## Where this lives in magos

| File                                            | Purpose                                              |
|--------------------------------------------------|------------------------------------------------------|
| `src/magos/routing/schema.py`                    | `Compress`, `CompressOptions`, `CompressMode` schema |
| `src/magos/routing/rewrites/compress/__init__.py` | `_apply_compress` dispatch entry point               |
| `src/magos/routing/rewrites/compress/token_mode.py` | Token-mode compression                              |
| `src/magos/routing/rewrites/compress/cache_mode.py` | `_apply_cache_aligner`, `_apply_compress_responses` |
| `src/magos/routing/rewrites/compress/model_limit.py` | `_resolve_model_limit` (registry + LiteLLM)        |
| `src/magos/routing/rewrites/compress/_preload.py` | `_preload_sentence_transformers` workaround          |
| `src/magos/routing/loader.py`                    | `Compress` listed in `_rewrites_touch_body`          |
| `src/magos/api/lifespan.py`             | Lifespan warmup hook + kompress backend override     |
| `tests/routing/rewrites/test_compress.py`        | Unit tests for both modes + endpoint scoping         |
| `tests/routing/rewrites/test_compress_registry.py` | `model_limit` resolution against registry           |
| `tests/routing/test_loader.py`                   | YAML round-trip + body-touch warning                 |
| `tests/api/test_lifespan.py`            | Lifespan warmup + kompress backend behaviour         |
| `docs/routing/grammar.md`                        | Operator-facing rewrite-op docs                      |
