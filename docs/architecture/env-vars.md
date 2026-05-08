# Environment variables

Resolution order (highest first) for the routing config path:

1. `--config <path>` CLI flag (and `--home` / `--models` for the data
   dir and registry-store paths)
2. `MAGOS_CONFIG_PATH` env var
3. `$MAGOS_HOME/magos.yaml` (default `~/.magos/magos.yaml`)

`MAGOS_HOME` is a **bootstrap-only env var**: it has no settings field
on `MagosSettings`. It anchors defaults for `MAGOS_CONFIG_PATH` and
`models.json`, and is the resolution base for relative registry paths
(not CWD, not the yaml file's parent). The resolution helpers live in
`config/settings.py` (`magos_home()`) and `config/loader.py`
(`resolve_models_path`).

| Variable                     | Default       | Purpose                                                |
|------------------------------|---------------|--------------------------------------------------------|
| `MAGOS_HOME`                 | `~/.magos`    | Data dir; anchors config and registry paths           |
| `MAGOS_CONFIG_PATH`          | `$MAGOS_HOME/magos.yaml` | Routing config YAML                       |
| `MAGOS_HOST`                 | (unset)       | Override `ingress.http.host` from yaml; yaml default is `127.0.0.1` |
| `MAGOS_PORT`                 | (unset)       | Override `ingress.http.port` from yaml; yaml default is `6246` |
| `MAGOS_MITM_ENABLED`         | (unset)       | Override `ingress.mitm.enabled`; yaml default is `false`            |
| `MAGOS_MITM_HOST`            | (unset)       | Override `ingress.mitm.host`; yaml default is `127.0.0.1`           |
| `MAGOS_MITM_PORT`            | (unset)       | Override `ingress.mitm.port`; yaml default is `6247`                |
| `MAGOS_MITM_INTERCEPT_HOSTS` | (unset)       | Comma-separated hosts; overrides `ingress.mitm.intercept_hosts`     |
| `MAGOS_LOG_LEVEL`            | `INFO`        | structlog level for `magos.*` loggers                  |
| `MAGOS_THIRD_PARTY_LOG_LEVEL`| `ERROR`       | Floor for every non-`magos.*` logger (uvicorn, litellm, httpx, transformers, ...). Raise to `WARNING`/`INFO`/`DEBUG` for debugging. Read directly via `os.environ` in `telemetry/logging.py`. |
| `MAGOS_LOG_JSON`             | `0`           | `1` flips renderer to JSON                             |
| `MAGOS_LOG_COLOR`            | auto-TTY      | `0`/`1` overrides TTY autodetect (read directly via `os.environ` in `telemetry/logging.py`; not a `MagosSettings` field, so `.env` loading does not apply) |
| `MAGOS_OTEL_ENABLED`         | `0`           | `1` ships OTel spans                                   |
| `MAGOS_OTEL_ENDPOINT`        | unset         | OTLP endpoint when OTel enabled                        |
| `MAGOS_KOMPRESS_BACKEND`     | `auto`        | `pytorch` forces PyTorch path (CUDA/MPS/CPU)           |
| `MAGOS_KOMPRESS_PRELOAD`     | `1`           | Preload Kompress weights at startup (only fires when a `compress` rule exists). Set to `0` for lazy on-demand load |
| `MAGOS_ACCESS_LOG`           | `1`           | `0` silences uvicorn access log                        |
| `MAGOS_METRICS_ENABLED`      | `0`           | `1` exposes Prometheus `/metrics`                      |
| `MAGOS_MODELS_PATH`          | `$MAGOS_HOME/models.json` | Override registry persistence path         |
