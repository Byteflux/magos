# Endpoint scope

| Endpoint                          | Field          | Compress support |
|-----------------------------------|----------------|------------------|
| `/v1/messages`                    | `messages`     | both modes       |
| `/v1/chat/completions`            | `messages`     | both modes       |
| `/v1/messages/count_tokens`       | `messages`     | both modes (useful: post-compression token preview) |
| `/v1/responses`                   | `instructions` | **`mode: cache` only** |
| `/v1/responses`                   | `input`        | unsupported (different shape from `messages`, no upstream Headroom path); `mode: token` silently no-ops |
| `/v1/responses/{id}` and friends  | n/a            | no-op (no body to compress)                         |

The Responses `instructions` string is wrapped as a synthetic
`[{"role": "system", "content": instructions}]` and fed to CacheAligner.
The aligner mutates the message's `content` in place; we read it back
and write it to `instructions`. No new messages are introduced. See
`_apply_compress_responses` in `rewrites/compress/cache_mode.py`.

Compressing `input` is not implemented. It would require a round-trip
converter for `input_text` / `message` / `function_call` / etc. items,
including atomicity preservation for `function_call` ↔
`function_call_output` pairs. Headroom's `HeadroomCallback`
(`integrations/litellm_callback.py`) explicitly filters
`call_type ∉ {completion, acompletion}` and only reads `data["messages"]`,
so there's no upstream conversion to mirror.

## Magos-Headroom mode terminology

Headroom uses its own "proxy mode" terminology in
`headroom/proxy/modes.py`:

- `PROXY_MODE_TOKEN`: prioritise compression (history may be
  rewritten).
- `PROXY_MODE_CACHE`: prioritise cache stability (freeze prior turns).

Magos mirrors these as the `mode: token | cache` switch on the
`compress` rewrite. The semantics differ in scope:

- Magos `mode: cache` runs **only** `CacheAligner` (no message-level
  changes).
- Magos `mode: token` runs the **full** pipeline (cache-aligned +
  routed + dropped if over budget).
