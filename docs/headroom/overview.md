# Overview

## What Headroom is

Two distinct subsystems shipped in one package:

1. **Compression pipeline** (`headroom.compress`): token-reduction
   transforms on `messages` lists.
2. **Cache optimizers** (`headroom.cache.*`): provider-specific cache
   helpers (Anthropic ephemeral breakpoints, OpenAI prefix caching,
   Google). Insert `cache_control` markers, track prefix hashes, score
   breakpoint placement.

Magos uses (1) directly. (2) is opt-in via `mode: cache` in the
`compress` rewrite, which runs only the `CacheAligner` transform, no
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

## Integration shapes considered

| Shape                                                | Verdict |
|------------------------------------------------------|---------|
| `headroom.transforms.TransformPipeline` direct       | **Adopted.** Owned by `magos.compression` (per-(config,provider) registry, inflation guard, eager warmup); the `Compress` rewrite primitive is the routing-layer caller. |
| `headroom.integrations.litellm_callback.HeadroomCallback` | **Rejected.** Implements LiteLLM's `CustomLogger.async_pre_call_hook`, which only fires when LiteLLM runs as a *proxy server*. Magos uses the LiteLLM SDK (`litellm.acompletion`, `litellm.anthropic_messages`, `litellm.aresponses`). The hook never fires in our architecture, verified by grep: `async_pre_call_hook` exists only under `litellm/proxy/`. |
| `headroom.proxy.handlers.*` (HeadroomProxy)          | **Rejected.** ~6,300 LOC of FastAPI handlers that re-implement provider routing, header forwarding, streaming. Stacking it under magos's mitmproxy + FastAPI duplicates routing. |
