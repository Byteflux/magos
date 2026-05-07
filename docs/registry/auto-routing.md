# Auto-routing

Explicit rules win. After the rules loop falls through, magos resolves
the inbound model in two passes:

1. **Namespaced lookup** — `registry.get(<inbound model id>)`. Matches
   when the client already sent a `<provider>/<raw_id>` key (e.g.
   `openrouter/anthropic/claude-sonnet-4-6`).
2. **Bare-id fallback** — when the namespaced lookup misses, magos
   collects every provider whose entry has `raw_id == <inbound model
   id>` (e.g. a request for `claude-sonnet-4-6` matches both
   `anthropic/claude-sonnet-4-6` and `openrouter/anthropic/claude-sonnet-4-6`
   if both are in the registry). One provider wins via the tie-break
   rule below; magos then re-keys to that provider's namespaced entry.

If both passes miss:

- `on_unknown_model: error` → 404.
- `on_unknown_model: passthrough` → hand the raw model string to
  LiteLLM and let it resolve via its bundled registry (works for names
  like `openai/gpt-4o`).

## Tie-breaking when multiple providers serve the same model

Resolution order: **explicit pin > `provider_order` > lexicographically
smallest provider**.

```yaml
# magos.yaml (top-level, alongside `providers:`)
provider_order:
  - openrouter   # preferred when no pin matches
  - anthropic
pins:
  claude-sonnet-4-6: anthropic   # always pick anthropic for this raw id
```

- `pins` is keyed by `raw_id` (no namespace prefix). A pin to a
  provider that doesn't actually serve the raw id is ignored — the
  resolution falls through to `provider_order` then lex.
- `provider_order` is a global preference list. The first listed
  provider that serves the raw id wins.
- If neither knob matches, the lex-smallest provider name wins
  (deterministic; not a quality signal).

The tie-break only fires for the bare-id fallback. A namespaced hit
in pass 1 short-circuits the whole machinery — operators who want a
specific provider should send the namespaced id and skip auto-routing
entirely.

## LiteLLM provider naming for openai-compatible upstreams

LiteLLM has no vultr-specific provider (verify with
`'vultr' in litellm.provider_list`). The same is true for most
openai-compatible third parties. LiteLLM does ship generic
openai-compatible shapes (`custom_openai`, `openai_like`,
`aiohttp_openai`) meant exactly for this case. Magos picks one by
stamping `litellm_id` with a litellm-known provider prefix:

| Adapter      | Default `litellm_provider` | Why                                                                                           |
|--------------|----------------------------|-----------------------------------------------------------------------------------------------|
| `openai`     | `openai`                   | OpenAI-specific provider; hits `api.openai.com` unless `api_base` is set.                     |
| `anthropic`  | `anthropic`                | Anthropic-specific provider.                                                                  |
| `openrouter` | `openrouter`               | OpenRouter-specific provider.                                                                 |
| `vultr`      | `custom_openai`            | No vendor-specific provider; use LiteLLM's generic openai-compatible shape, which requires explicit `api_base` and won't silently fall back to `api.openai.com` + `OPENAI_API_KEY` the way bare `openai` would. |
| `noop`       | n/a                        | Manual-only provider (no upstream discovery). Set `litellm_provider` explicitly per entry — no default — and prefer `custom_openai` for openai-compatible hosts. |

Picking `openai` for a non-OpenAI host is the common footgun: the call
succeeds in flight but lands on `api.openai.com` with `OPENAI_API_KEY`,
returning a misleading 401 "Incorrect API key provided: sk-proj-…":
looks like an auth bug, is really a routing bug.

When adding a new openai-compatible adapter for a host with no
vendor-specific LiteLLM provider, default `_DEFAULT_LITELLM_PROVIDER`
to `custom_openai` and require operators to set `base_url` +
`api_key_env` so the dispatcher can pass both to LiteLLM.
