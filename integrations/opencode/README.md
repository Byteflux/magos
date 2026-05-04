# magos · OpenCode plugin

Registers every non-deprecated model from a running magos instance with
[OpenCode](https://opencode.ai) under a single `magos` provider that
speaks the OpenAI Chat/Responses wire shape.

The plugin pulls the live registry from `GET /admin/registry`, so it
sees the same context windows, output limits, costs, and modalities
that magos uses for routing.

## Install

OpenCode (`opencode.json` `"plugin"` array) only accepts npm specs, so
the simplest install is to drop the source file into one of OpenCode's
plugin directories:

```bash
# global (loads in every project)
ln -s ~/Projects/magos/integrations/opencode/magos.ts \
      ~/.config/opencode/plugins/magos.ts

# or project-local
mkdir -p .opencode/plugins
ln -s ~/Projects/magos/integrations/opencode/magos.ts \
      .opencode/plugins/magos.ts
```

OpenCode runs on Bun, which imports `.ts` directly — no build step.

## Configure OpenCode

The default install requires no `opencode.json` changes — the plugin
injects a `provider.magos` block at startup, and the SDK package
(`@ai-sdk/openai-compatible`) plus base URL come from each registered
`ModelV2`.

To point the plugin at a non-default magos host, add an explicit
provider block:

```json
{
  "provider": {
    "magos": {
      "options": { "baseURL": "http://192.168.10.100:6246/v1" }
    }
  }
}
```

The plugin reads `options.baseURL` from this block, so both
`/admin/registry` lookups and SDK wire calls hit the same host. The
trailing `/v1` is required by the SDK and is stripped internally for
the registry fetch.

`MAGOS_BASE_URL` (env var, no `/v1` suffix) is honored as a fallback
when `options.baseURL` isn't set. Resolution order:

1. `provider.magos.options.baseURL` from `opencode.json`
2. `$MAGOS_BASE_URL` environment variable
3. `http://localhost:6246` (default)

## How it works

1. On startup, OpenCode loads `magos.ts` and calls its `server` hook,
   which returns a `config` callback.
2. The `config` callback fires before OpenCode reads `cfg.provider`,
   so it can both default missing fields and inject the model dict
   directly into `cfg.provider.magos.models`.
3. It resolves the magos host (priority: `provider.magos.options.baseURL`
   → `MAGOS_BASE_URL` → `http://localhost:6246`), `fetch`es
   `${host}/admin/registry` with a 3s timeout, and translates each entry
   to OpenCode's config-model shape.
4. OpenCode then parses `cfg.provider.magos` (now fully populated)
   through its config-providers loop and registers magos as a custom
   provider, models and all.
5. If magos isn't reachable, the plugin logs a single warning and skips
   model injection — OpenCode still starts cleanly, but the magos
   provider will have no models.

Model ids are namespaced (`<magos-provider>/<raw-id>`, e.g.
`openrouter/x-ai/grok-4.3`) and surface in the OpenCode model picker as
`magos/openrouter/x-ai/grok-4.3`. The wire-level `model` field sent to
magos is the same namespaced id (`openrouter/x-ai/grok-4.3`), which is
what `magos.yaml` rules and the registry-driven auto-router match
against.

> **Why not `provider.models()`?** That hook exists, but as of opencode
> 1.14.33 it's silently skipped for any provider not already in the
> `models.dev` catalog (see `provider/provider.ts:1153` in the opencode
> source). magos isn't in the catalog, so the only way to register
> models is through `cfg.provider.magos.models` — which is what the
> `config` hook populates.

## Verify

Run `opencode models magos` to confirm the plugin is registering
models. You should see hundreds of `magos/<provider>/<raw-id>` entries.
If the list is empty, check that magos is running on `MAGOS_BASE_URL`
and that `curl ${MAGOS_BASE_URL}/admin/registry` returns 200.

## Environment

| Variable | Default | Purpose |
|---|---|---|
| `MAGOS_BASE_URL` | `http://localhost:6246` | Fallback magos host when `provider.magos.options.baseURL` isn't set in `opencode.json`. No `/v1` suffix. |

## Static-block fallback

If the plugin hook is bypassed in your OpenCode build (some versions
skip `provider.models()` for providers that aren't in the `models.dev`
catalog — see [sst/opencode#25630](https://github.com/sst/opencode/issues/25630)),
fall back to a static `models` block in `opencode.json` until the
upstream fix lands:

```bash
magos models list --format json | jq '...' > magos-models.json
# paste into opencode.json under provider.magos.models
```

The plugin code itself doesn't need to change.

## Type-check

For editor / CI:

```bash
cd integrations/opencode
bun install            # pulls @opencode-ai/plugin + @opencode-ai/sdk type stubs
bunx tsc --noEmit
```
