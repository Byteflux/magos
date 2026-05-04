# Operator CLI

The `magos` command is a small Typer app installed by
`[project.scripts]` in `pyproject.toml`. The CLI lives in
`src/magos/cli/`; `python -m magos` and `magos` resolve to the same
entrypoint (`magos.cli.app:main`).

```
magos                       # serve (default)
magos serve                 # explicit
magos models <verb>         # registry inspection / management
magos --version             # print version and exit
magos --config <path> ...   # override the config file
```

## Top-level options

| Flag         | Effect                                                          |
|--------------|-----------------------------------------------------------------|
| `--config`   | Path to `magos.yaml`. Overrides `MAGOS_CONFIG_PATH` and the `$MAGOS_HOME/magos.yaml` default. |
| `--version`  | Print `magos <version>` and exit.                               |
| `-h` / `--help` | Show help.                                                   |

`--config` is on the root command; subcommands inherit. Order matters:
`magos --config /etc/x.yaml models list` works,
`magos models --config /etc/x.yaml list` does not.

## `magos serve`

Run the FastAPI server (and the optional embedded mitmproxy ingress
when `server.ingress.enabled` is true in `magos.yaml`). This is the
default when no subcommand is given.

```bash
magos                          # equivalent to "magos serve"
magos serve --port 9000        # override MAGOS_PORT and yaml
magos serve --host 0.0.0.0     # listen on all interfaces
```

| Flag         | Effect                                                            |
|--------------|-------------------------------------------------------------------|
| `--host`     | HTTP listen host. Stamps `MAGOS_HOST`; overrides yaml + env.      |
| `--port`     | HTTP listen port. Stamps `MAGOS_PORT`; overrides yaml + env.      |

Bind precedence (highest first): `--host` / `--port` flags >
`MAGOS_HOST` / `MAGOS_PORT` env > `server.host` / `server.port` in
yaml > defaults (`127.0.0.1:8000`).

The CLI bootstrap (logging + tracing config + the
`server.bootstrapping` log event) happens here, then control hands
off to the orchestrator in `magos.serve`. See
[`docs/architecture.md`](architecture.md) for the request lifecycle
and process topology.

## `magos models`

Inspect and manage the model registry. Read commands fall back to the
on-disk `models.json` when the server isn't reachable. Mutating
commands require the running server.

```bash
magos models list                    # in-memory state from server
magos models list --from-disk        # bypass server, read models.json
magos models list --format json      # machine-readable

magos models show <namespaced-id>
magos models show <namespaced-id> --from-disk

magos models refresh                 # trigger refresh on all providers
magos models refresh --provider X    # scope to one provider
magos models prune                   # sweep past-grace deprecated entries

magos models discover --provider X --dry-run
```

Full reference + the registry lifecycle is in
[`docs/registry.md`](registry.md). The CLI hits
`/admin/registry/{,refresh,prune}` endpoints when talking to the
running server.

## Environment variables

Settings (read from the process env, optionally via `.env`):

| Variable                    | Default                       | Notes                                      |
|-----------------------------|-------------------------------|--------------------------------------------|
| `MAGOS_HOME`                | `~/.magos`                    | Bootstrap-only; anchors yaml + models.json defaults. |
| `MAGOS_CONFIG_PATH`         | `$MAGOS_HOME/magos.yaml`      | Routing config. CLI `--config` wins.       |
| `MAGOS_MODELS_PATH`         | yaml `registry.models_path` or `$MAGOS_HOME/models.json` | Override registry persistence path.        |
| `MAGOS_HOST`                | yaml `server.host` or `127.0.0.1` | HTTP listen host. CLI `--host` wins.   |
| `MAGOS_PORT`                | yaml `server.port` or `8000`  | HTTP listen port. CLI `--port` wins.       |
| `MAGOS_LOG_LEVEL`           | `INFO`                        | structlog filter level.                    |
| `MAGOS_LOG_JSON`            | `0`                           | `1` to emit JSON instead of structured text. |
| `MAGOS_LOG_COLOR`           | auto (TTY)                    | `0`/`1` to force off/on regardless of TTY. |
| `MAGOS_ACCESS_LOG`          | `1`                           | One structlog line per HTTP request.       |
| `MAGOS_OTEL_ENABLED`        | `0`                           | Ship OTLP spans.                           |
| `MAGOS_OTEL_ENDPOINT`       | OTel SDK default              | OTLP HTTP endpoint.                        |
| `MAGOS_METRICS_ENABLED`     | `0`                           | Mount Prometheus `/metrics` endpoint.      |
| `MAGOS_KOMPRESS_BACKEND`    | `auto`                        | `auto` or `pytorch`. `pytorch` forces the GPU-friendly path. |
| `MAGOS_KOMPRESS_PRELOAD`    | `1`                           | Background-load Kompress weights at startup. |

Provider API keys (read by routing rules' `api_key_env` setting) are
not part of `MagosSettings` — they're declared per-rule in
`magos.yaml` and read at request time. Common ones:

| Variable                | Used by                                         |
|-------------------------|-------------------------------------------------|
| `ANTHROPIC_API_KEY`     | Anthropic provider rules + LiteLLM auto-routing |
| `OPENAI_API_KEY`        | OpenAI provider rules + LiteLLM auto-routing    |
| `OPENROUTER_API_KEY`    | OpenRouter provider rules                       |
| `VULTR_API_KEY`         | Vultr provider rules                            |

The full set of inert (removed) env vars is logged at startup as
`config.removed_env_var` warnings; see
[`docs/routing.md`](routing.md#inert-env-vars).

## Exit codes

| Code | Meaning                                                              |
|------|----------------------------------------------------------------------|
| `0`  | Success.                                                             |
| `1`  | Subcommand-specific failure (unknown model id, partial refresh failure). |
| `2`  | Operator error (unreachable server for a mutating command, unknown provider, server returned an error). |

## See also

- [`docs/architecture.md`](architecture.md) — request lifecycle,
  process topology, env-var resolution.
- [`docs/routing.md`](routing.md) — `magos.yaml` grammar.
- [`docs/registry.md`](registry.md) — registry config, lifecycle, and
  per-command behaviour for `magos models`.
- [`docs/ingress.md`](ingress.md) — embedded mitmproxy ingress for
  clients that change behaviour when their `BASE_URL` is overridden.
- [`docs/deployment.md`](deployment.md) — Docker / compose deployment.
