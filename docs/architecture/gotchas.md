# Subtleties worth not forgetting

- **Routing rules always beat the registry.** Auto-routing is a
  fallback for unmatched requests, not a parallel layer.
- **`models.json` has one writer (Refresher).** Don't add direct writes.
- **`body_dirty` is mandatory for body-mutating rewrites.** Forgetting
  it sends pre-rewrite bytes through passthrough.
- **Passthrough is byte-exact for cache + OAuth reasons.** No
  normalisation. No LiteLLM round-trip.
- **`litellm.drop_params=True` is global.** Suspect this first when a
  param vanishes.
- **mitmproxy is opt-in for ingress.** When `ingress.mitm.enabled`
  is false (the default), mitmproxy is completely dormant: no
  listener, no addon hooks running. When enabled, it terminates TLS
  for allowlisted hosts and rewrites to FastAPI loopback; routing
  itself still happens in FastAPI.
- **`sentence_transformers` preload in conftest is load-bearing**
  (Windows-only crash, but the preload is unconditional so CI/Linux
  pays nothing).
- **Headroom `_is_onnx_available` is monkey-patched at startup** when
  `MAGOS_KOMPRESS_BACKEND=pytorch`. Looks weird, is intentional.
- **Anthropic OAuth (`sk-ant-oat`) auth shape lives in two places**:
  `egress/auth.py` (proxy-side injection) and
  `registry/discovery/anthropic.py` (discovery). Keep them in sync.
- **Header blocking is three-level**: ingress inbound, pre-LiteLLM
  body shape, and pre-LiteLLM auth (conditional on rule-resolved
  `api_key`). All three must be checked when a header isn't reaching
  the provider.
