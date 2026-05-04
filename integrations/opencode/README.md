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

The plugin only supplies the model list. The wire transport
(`@ai-sdk/openai-compatible` pointed at magos) goes in
`opencode.json` (or `opencode.jsonc`):

```json
{
  "provider": {
    "magos": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Magos",
      "options": { "baseURL": "http://localhost:6246/v1" }
    }
  }
}
```

If your magos isn't on the default port, set `MAGOS_BASE_URL` in the
shell that launches OpenCode and update `options.baseURL` to match.

## How it works

1. On startup, OpenCode loads `magos.ts` and calls its `server` hook.
2. The hook returns a `provider` hook with `id: "magos"`.
3. OpenCode invokes `provider.models()`, which `fetch`es
   `${MAGOS_BASE_URL}/admin/registry` (3s timeout) and maps each entry
   to OpenCode's `ModelV2` shape.
4. If magos isn't reachable, the plugin logs a single warning and
   returns no models — OpenCode still starts cleanly.

Model ids are namespaced (`<magos-provider>/<raw-id>`, e.g.
`openrouter/x-ai/grok-4.3`) and surface in the OpenCode model picker as
`magos/openrouter/x-ai/grok-4.3`. The wire-level `model` field sent to
magos is the same namespaced id (`openrouter/x-ai/grok-4.3`), which is
what `magos.yaml` rules and the registry-driven auto-router match
against.

## Verify

Run `opencode models magos` to confirm the plugin is registering
models. You should see hundreds of `magos/<provider>/<raw-id>` entries.
If the list is empty, check that magos is running on `MAGOS_BASE_URL`
and that `curl ${MAGOS_BASE_URL}/admin/registry` returns 200.

## Environment

| Variable | Default | Purpose |
|---|---|---|
| `MAGOS_BASE_URL` | `http://localhost:6246` | Base URL of the running magos instance. |

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
