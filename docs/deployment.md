# Deployment

The repo ships a `Dockerfile` and `compose.yaml` aimed at GPU-backed
deployments where Headroom's Kompress runs on PyTorch+CUDA. CPU-only
deployments work too — just drop the `deploy.resources.reservations`
block from compose and the GPU-extra install is harmless.

## What's in the image

- Python 3.12 (slim).
- magos installed from the repo via `uv sync --extra gpu --no-dev`.
  The `gpu` extra pulls CUDA-enabled PyTorch wheels; swap to
  `--extra cpu` when building for a CPU-only host.
- `magos.example.yaml` copied to `/etc/magos/magos.yaml` as a starter
  config (compose mounts your real one over it).
- `CMD ["magos", "serve"]`.

Defaults baked into the image (override via env or `--port` / `--host`):

| Env var                 | Image default                      |
|-------------------------|------------------------------------|
| `MAGOS_HOST`            | `0.0.0.0`                          |
| `MAGOS_PORT`            | `2570`                             |
| `MAGOS_CONFIG_PATH`     | `/etc/magos/magos.yaml`            |
| `MAGOS_MODELS_PATH`     | `/var/lib/magos/models.json`       |
| `MAGOS_KOMPRESS_BACKEND`| `pytorch`                          |
| `MAGOS_LOG_COLOR`       | `1`                                |

The GPU image picks `pytorch` over `auto` because Headroom prefers ONNX
Runtime when both are present, and ONNX doesn't see CUDA in this
environment. Forcing PyTorch routes Kompress through the CUDA-enabled
torch wheel installed in the image. See
[`docs/headroom.md`](headroom.md) for backend details.

## Compose

```yaml
# compose.yaml
services:
  magos:
    build: .
    image: ghcr.io/byteflux/magos
    ports:
      - "2570:2570"
    env_file:
      - .env
    volumes:
      - ${USERPROFILE:-${HOME:-.}}/.cache/huggingface:/root/.cache/huggingface
      - ./magos.yaml:/etc/magos/magos.yaml:ro
      - magos-state:/var/lib/magos
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

Volume layout:

- `~/.cache/huggingface` (host) → `/root/.cache/huggingface` (container).
  HF model weights persist across restarts; first launch downloads
  Kompress weights, subsequent launches reuse them.
- `./magos.yaml` (host) → `/etc/magos/magos.yaml` (container, read-only).
  Edit on the host; the container picks up the updated file on the
  next restart. The image's bundled `magos.example.yaml` is overlaid by
  this mount when the file exists on the host.
- `magos-state` named volume → `/var/lib/magos`.
  Holds `models.json` (the registry's discovered-model cache). Persists
  across container rebuilds. Safe to delete; the registry rebuilds via
  live discovery on next boot.

`.env` (in the same directory as `compose.yaml`) is the standard place
for provider API keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.) —
loaded into the container's process env via `env_file`.

## GPU prerequisites

For the compose `nvidia` reservation to work:

1. NVIDIA driver installed on the host.
2. NVIDIA Container Toolkit installed and configured for your container
   runtime (`nvidia-ctk runtime configure --runtime=docker` on Linux,
   or the WSL2-backed Docker Desktop GPU support on Windows).

Verify with `docker run --rm --gpus all nvidia/cuda:12.4.0-base nvidia-smi`
before bringing magos up.

For CPU-only: drop the `deploy:` block from compose and rebuild with
`--extra cpu` in the Dockerfile (or just leave `gpu` and let CUDA be
unused — the wheel is fatter but works).

## CPU-only build

Edit the Dockerfile to swap `gpu` → `cpu`:

```dockerfile
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra cpu --no-dev --no-editable
```

…and remove the `deploy.resources.reservations` block from compose.
Set `MAGOS_KOMPRESS_BACKEND=auto` (or unset it) so Headroom picks ONNX
where available.

## Environment-variable layering

The image bakes opinionated defaults; compose's `.env` overrides them
per-deploy; CLI flags override the env. Resolution order, highest
first:

1. CLI flags (`--host`, `--port`, `--config`).
2. `.env` / `env_file` / shell env.
3. `Dockerfile` `ENV` defaults.
4. yaml defaults from `magos.yaml`.
5. magos's hardcoded fallbacks (`127.0.0.1:8000`, etc.).

Notable: `MAGOS_HOST=0.0.0.0` in the Dockerfile is what makes the
container reachable from the host's mapped port. Don't override it
back to loopback unless you know what you're doing.

## Loop hazard with mitmproxy ingress

If you enable `ingress.mitm.enabled: true` in yaml, the container
runs both FastAPI (default 2570) and mitmproxy (default 8080). The
client `HTTPS_PROXY` setting is yours to manage on the host;
[`docs/ingress.md`](ingress.md) covers the loop-hazard caveat in
detail. Map both ports if you intend to use the proxy from outside the
container.

## Health and observability

- `MAGOS_METRICS_ENABLED=1` mounts `/metrics` (Prometheus). Scrape
  from the same port as the API.
- Logs go to stderr in structlog format; set `MAGOS_LOG_JSON=1` for
  machine-readable output suitable for a log aggregator.
- `MAGOS_OTEL_ENABLED=1` ships OTLP spans (set `MAGOS_OTEL_ENDPOINT`
  to your collector).

There is no dedicated `/healthz` endpoint today; FastAPI returns 200
on `/openapi.json` and any unmatched path on a configured endpoint
returns the routing-shaped 404. Use one of those for liveness probes.

## See also

- [`docs/cli.md`](cli.md) — full env-var table, CLI flags.
- [`docs/architecture.md`](architecture.md) — request lifecycle.
- [`docs/headroom.md`](headroom.md) — Kompress backend selection,
  model preload behaviour.
- [`docs/ingress.md`](ingress.md) — mitmproxy ingress setup +
  loop-hazard.
