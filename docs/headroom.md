# Headroom integration notes

Reference for how Headroom works under magos and the non-obvious findings
behind the integration in `src/magos/routing/rewrites/compress.py`.
Verified against `headroom-ai==0.10.16`.

| Topic | Contents |
|-------|----------|
| [Overview](headroom/overview.md) | What Headroom is, Anthropic prompt caching accurately, integration shapes considered |
| [Pipeline](headroom/pipeline.md) | The compression pipeline: CacheAligner, ContentRouter, Kompress, IntelligentContextManager |
| [model_limit resolution](headroom/model-limit.md) | How magos resolves `model_limit`, plus `CompressConfig` defaults that matter |
| [Failure modes](headroom/failure-modes.md) | Failure semantics and pipeline init cost |
| [Backend](headroom/backend.md) | Forcing the Kompress backend (`MAGOS_KOMPRESS_BACKEND`) |
| [Endpoint scope](headroom/endpoint-scope.md) | Per-endpoint compress support and Magos-Headroom mode terminology |
| [Gotchas](headroom/gotchas.md) | Subtleties worth not forgetting and where this lives in magos |
