# YAML grammar

```yaml
pre_rewrites: []          # global rewrites; optional. Each entry is either
                          # a Rewrite (unconditional) or a guarded group:
                          #   { match: <expr>, rewrites: [<Rewrite>, ...] }
                          # Guarded groups apply only when match is true at
                          # that point in the pre-rewrite chain.
rules:                    # required, at least one
  - name: human-readable  # optional; appears in route.matched logs
    match: <expr>
    rewrites: []          # per-rule post-rewrites; optional
    target:
      provider: <string>  # required
      gateway: translate | passthrough
      base_url: <url>     # required when gateway=passthrough
      api_key_env: <NAME> # optional
      auth_header: <shape># optional; bearer | x-api-key. Defaults to
                          # x-api-key for provider: anthropic, bearer
                          # otherwise. Only consulted when injecting
                          # api_key_env in passthrough gateway; explicit
                          # client headers always pass through verbatim.
```

count_tokens calls go through `litellm.acount_tokens`, which auto-selects
between an in-process tokenizer and the provider's native count-tokens
endpoint based on the model id. There is no separate `count_tokens_mode`
knob; declare a regular `gateway: translate` rule for `/v1/messages/count_tokens`.

## Match expressions

Atoms (each is a single-key dict):

| Atom          | Shape                                                          | Matches against         |
|---------------|----------------------------------------------------------------|-------------------------|
| `model`       | `{ model: <matcher> }`                                         | `body.model` (string)   |
| `header`      | `{ header: { name: <matcher>, value: <matcher> } }`            | any inbound header pair |
| `endpoint`    | `{ endpoint: <matcher> }`                                      | `/v1/messages`, `/v1/messages/count_tokens`, `/v1/chat/completions`, `/v1/responses`, `/v1/responses/{id}`, `/v1/responses/{id}/input_items` |
| `jq`          | `{ jq: "<expr>" }`                                             | parsed body (truthy)    |
| `model_field` | `{ model_field: { field: <name>, op: <op>, value: <value> } }` | a registry-resolved field on the inbound model (see [matchers](./matchers.md)) |

`<matcher>` is exactly one of:

- `{ literal: "x" }`: exact equality, case-sensitive
- `{ glob: "x*" }`: fnmatch, case-sensitive
- `{ regex: "^x" }`: `re.fullmatch`, no implicit flags

Combinators:

- `{ all_of: [<expr>, ...] }`: every child must match
- `{ any_of: [<expr>, ...] }`: at least one child must match
- `{ not: <expr> }`: child must not match

A bare atom at the top of `match` is shorthand for a single-atom expression.

## Rewrite ops

Each is a single-key dict applied in list order:

| Op             | Shape                                     | Effect                                 |
|----------------|-------------------------------------------|----------------------------------------|
| `set_model`    | `{ set_model: "x" }`                      | replace `body.model`; flips body_dirty |
| `set_header`   | `{ set_header: { name: ..., value: ... }}`| insert or overwrite (case-insensitive) |
| `add_header`   | `{ add_header: { name: ..., value: ... }}`| insert only if absent                  |
| `remove_header`| `{ remove_header: "name" }`               | drop if present                        |
| `jq_patch`     | `{ jq_patch: "<expr>" }`                  | result replaces body; must be a JSON object |
| `compress`     | `{ compress: { ... } }`                   | run Headroom compression on `messages`; flips body_dirty |

`jq_patch`, `set_model`, and `compress` mark the request body as dirty.
Under `gateway: passthrough`, a dirty body forces re-serialisation,
breaking prompt-cache byte-exactness; the loader debug-logs each
offending rule at startup (event `routing.passthrough_body_touch`).

### `set_model` and the registry

For `gateway: translate`, the engine resolves the dispatch model id in this
order before handing it to LiteLLM:

1. Literal registry hit on the body model, e.g.
   `set_model: vultr/Qwen/Qwen3.5-...` looks up that exact key and
   substitutes the entry's `litellm_id` (`custom_openai/Qwen/...`).
2. Registry hit on `<action.provider>/<model>`, e.g. with
   `provider: vultr` in the action, `set_model: Qwen/Qwen3.5-...`
   resolves the same entry without the prefix.
3. Otherwise, the model is passed through as-is if it already contains
   `/`, or prefixed with `<action.provider>/` if bare.

This matters for openai-compatible third parties (Vultr, hosted vLLM,
etc.) that route through the `custom_openai` provider in LiteLLM: the
magos namespace (`vultr/`) and LiteLLM's dispatch id (`custom_openai/`)
diverge, and registry consultation reconciles them. Configure providers
under `providers:` so the registry knows the mapping; see
[registry.md](../registry.md).

### `compress`

Runs Headroom against `body.messages`. Two modes:

- `mode: token` (default): full pipeline (CacheAligner + ContentRouter
  + IntelligentContext). Messages may be rewritten or dropped. Maximises
  token savings.
- `mode: cache`: CacheAligner only. Extracts dynamic content (dates,
  whitespace) from system prompts so the prefix is byte-stable across
  requests. Does not touch user/assistant messages. Improves provider
  prompt-cache hit rate without changing semantics.

Endpoint scope:

- `/v1/messages`, `/v1/messages/count_tokens`, `/v1/chat/completions`:
  full support for both modes against `body['messages']`.
- `/v1/responses`: `mode: cache` only, against `body['instructions']`.
  Token-mode compression of the `input` field is unsupported (different
  shape, no upstream Headroom path); `mode: token` silently no-ops here.
- `/v1/responses/{id}` family (retrieve / cancel / list input items):
  no-op (no body to compress).

Failure mode: Headroom fails open internally. On any compression error
the original messages pass through and an OTel metric is recorded.

All `CompressConfig` knobs are surfaced verbatim, plus an explicit
`model_limit` override and the magos-side CCR (reversible-compression)
toggles:

```yaml
rewrites:
  - compress:
      mode: token              # token | cache
      compress_user_messages: false
      compress_system_messages: true
      protect_recent: 4        # last N messages untouched
      protect_analysis_context: true
      target_ratio: null       # null = aggressive default
      min_tokens_to_compress: 250
      kompress_model: null     # HF model id, or "disabled"
      model_limit: null        # null = auto-detect via litellm; or e.g. 128000

      # ContentRouter / IntelligentContext knobs (smart_routing path)
      smart_routing: true        # false = legacy SmartCrusher-only
      code_aware: false          # AST-aware code compression (needs tree-sitter)
      intelligent_context: true  # false = RollingWindow (last-N turns)
      keep_last_turns: 4         # context manager preserves last N verbatim

      # CCR (reversible compression) injection toggles
      ccr_enabled: true              # false disables tool / instruction injection
      ccr_inject_tool: true          # false if a client distributes the tool via MCP
      ccr_inject_instructions: true  # auto-skipped while prefix-cache freeze count > 0
```

#### CCR (reversible compression)

When `ccr_enabled: true` (default) and post-compression messages
contain Headroom compression markers, the `compress` rewrite injects
the `headroom_retrieve` tool definition into `body.tools` and a
short system-message instruction block telling the model how to use
it. Egress dispatch then wraps the response (streaming and
non-streaming) so any `headroom_retrieve` tool call from the model
is intercepted, the original content is restored, and the request
is re-run transparently.

- `ccr_enabled: false` — disable CCR entirely for this rule.
  Compression still runs and emits markers, but no tool / instructions
  are injected and no response interception occurs.
- `ccr_inject_tool: false` — skip tool injection when a client already
  distributes the `headroom_retrieve` tool via MCP and re-injection
  would duplicate it.
- `ccr_inject_instructions: false` — skip instruction injection.
  Instruction injection is *also* automatically skipped whenever the
  prefix-cache freeze count is non-zero (preserves cache hits),
  regardless of this flag.

See [`docs/headroom/pipeline.md`](../headroom/pipeline.md) for the
end-to-end CCR flow and the per-session prefix-cache tracker.

`model_limit` controls when Headroom's IntelligentContextManager fires
(over-budget message dropping) and how aggressively ContentRouter
scales compression. By default magos calls `litellm.get_model_info`
on the dispatch model, reads `max_input_tokens`, and falls back to
200000 for unknown models. Set explicitly to leave headroom for
output budgets, force earlier compression for cost reasons, or pin a
value for custom models that LiteLLM doesn't recognise.

Startup: when any rule uses `compress`, the FastAPI lifespan hook warms
Headroom's pipeline (tokenizer + transform init) so first-request
latency is amortised.
