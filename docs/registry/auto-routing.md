# Auto-routing

Explicit rules win. After the rules loop falls through, magos attempts
`registry.get(<inbound model id>)`:

- Hit → synthesize a translate-mode `Action` for the entry's provider,
  hand the entry's `litellm_id` to the dispatcher.
- Miss + `on_unknown_model: error` → 404.
- Miss + `on_unknown_model: passthrough` → hand the raw model string to
  LiteLLM and let it resolve via its bundled registry (works for names
  like `openai/gpt-4o`).

When multiple providers serve the same logical model, tie-breaking:
explicit pin > `provider_order` > lexicographically smallest provider.

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

Picking `openai` for a non-OpenAI host is the common footgun: the call
succeeds in flight but lands on `api.openai.com` with `OPENAI_API_KEY`,
returning a misleading 401 "Incorrect API key provided: sk-proj-…":
looks like an auth bug, is really a routing bug.

When adding a new openai-compatible adapter for a host with no
vendor-specific LiteLLM provider, default `_DEFAULT_LITELLM_PROVIDER`
to `custom_openai` and require operators to set `base_url` +
`api_key_env` so the dispatcher can pass both to LiteLLM.
