# CLI

```bash
magos models list                     # in-memory state from server
magos models list --from-disk         # bypass server, read models.json
magos models list --format json       # machine-readable

magos models show openrouter/anthropic/claude-sonnet-4-6
magos models show <id> --from-disk

magos models refresh                  # all providers
magos models refresh --provider openrouter

magos models prune                    # sweep past-grace deprecated entries

magos models discover --provider openrouter --dry-run
```

Every subcommand accepts `--config <path>` to point at a non-default
yaml; precedence is `--config` > `MAGOS_CONFIG_PATH` > the
`~/.magos/magos.yaml` default.

`list` and `show` fall back to disk if the server isn't reachable.
`refresh` and `prune` require the server to be running and hit
`POST /admin/registry/{refresh,prune}`.

## Public listing: `GET /v1/models`

The registry is also surfaced to API clients via `GET /v1/models`. The
response shape is content-negotiated: requests carrying
`anthropic-version` or `x-api-key` get the Anthropic shape (`{data,
has_more, first_id, last_id}` with `type: "model"`, `display_name`,
`created_at`); everything else gets the OpenAI shape (`{object: "list",
data: [{id, object: "model", created, owned_by}]}`). Entries are
filtered to live (non-deprecated) records and sorted by `namespaced_id`
(`<provider>/<raw_id>`). When no providers are configured (registry
feature dormant), the list is empty rather than 404 so clients can
probe unconditionally.
