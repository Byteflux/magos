# Headers and auth

## Auth-header injection

`magos.egress.auth` injects an outbound auth header iff:

- `action.mode == "passthrough"` AND
- The inbound request lacks both `Authorization` and `x-api-key`, AND
- The matched rule's `action.api_key_env` resolves to a non-empty env
  var.

(Translate mode reads `action.api_key_env` separately and hands it to
LiteLLM as the `api_key` kwarg; it does not write headers.)

**Resolution order for the header shape (highest first):**

1. **OAuth detection**: `provider: anthropic` rule whose
   `api_key_env` value starts with `sk-ant-oat` →
   `Authorization: Bearer <token>` + `anthropic-beta: oauth-2025-04-20`
   (overrides everything; api.anthropic.com 401s on `x-api-key` for
   that credential class). The provider guard is required: a
   non-anthropic rule whose env var happens to start with that prefix
   does not get OAuth headers.
2. **Per-rule `action.auth_header` override**: explicit
   `x-api-key` or `bearer` value on the rule.
3. **Provider default**: `provider: anthropic` → `x-api-key`,
   everything else → `Authorization: Bearer`.

OAuth-token detection lives in `egress/auth.py` and the registry-side
discovery counterpart in `registry/discovery/anthropic.py`. If you
change one, change both or the registry's discovery call will fail
against an OAuth-only account.

## Header forwarding is multi-stage

| Stage                         | What's blocked                                           | Why                                              |
|-------------------------------|----------------------------------------------------------|--------------------------------------------------|
| Ingress inbound (`ingress/http/headers.py`) | RFC 7230 hop-by-hop + `host` / `content-length` / `content-encoding` / `accept-encoding` | Don't propagate transport-layer junk             |
| Pre-LiteLLM body shape (`egress/translate/payload.py`) | `content-type` / `content-length` / `content-encoding` / `accept-encoding` | LiteLLM regenerates these; overriding causes "unexpected keyword argument" errors at the SDK boundary |
| Pre-LiteLLM auth (`egress/translate/payload.py`) | `authorization` / `x-api-key` (**only when** the rule's `api_key` was resolved) | Stops the inbound bearer from leaking into `extra_headers` and overriding the operator-chosen upstream key |
| Pre-passthrough               | nothing additional                                       | Byte-exact forwarding (cache hashes)             |

If a header you expect to see at the provider isn't arriving, check
all three stages.
