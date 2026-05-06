# Ingress: transparent HTTPS proxy via mitmproxy

magos's normal mode of operation has clients pointing directly at the
FastAPI listener (e.g. `ANTHROPIC_BASE_URL=http://localhost:6246`).
For clients that change behavior when their base URL is overridden
(notably **Claude Code**, which alters auth/feature handling when
`ANTHROPIC_BASE_URL` is set), that mode is unusable.

The ingress proxy is the workaround. magos runs an embedded
`mitmproxy` listener alongside FastAPI in the same process. A client
pointed at `HTTPS_PROXY=http://127.0.0.1:6247` keeps thinking it's
talking to `api.anthropic.com`; mitmproxy terminates TLS, the addon
rewrites the decrypted request to magos's loopback FastAPI listener,
and the response flows back transparently.

## When to use it

- Claude Code (and any other client whose behavior shifts when
  `BASE_URL` env vars are set) needs to reach magos.
- You want a single magos process responsible for both the routing
  layer and the transparent intercept, on one port pair, one config.

For native-`base_url`-aware clients (most LLM SDKs, including
`anthropic`, `openai`, `litellm`), you don't need the ingress proxy:
just point them at `http://localhost:6246`.

## Architecture

```
┌──────────── single magos process ─────────────────┐
│  asyncio event loop                               │
│  ├── FastAPI (uvicorn)        → listens :6246     │
│  └── mitmproxy (DumpMaster)   → listens :6247     │
│       ├── MagosIngressAddon   (rewrite host)      │
│       ├── MagosObserverAddon  (egress logging)    │
│       └── structlog bridge    (uniform log shape) │
└───────────────────────────────────────────────────┘
```

Per request: client speaks HTTPS to mitmproxy on 6247 with
SNI=`api.anthropic.com`. The addon's `tls_clienthello` hook checks the
SNI against the allowlist; if it matches, mitmproxy terminates TLS
with a CA-signed cert for that hostname. The decrypted request hits
the addon's `request` hook, which rewrites `host`/`port`/`scheme` to
`127.0.0.1:6246` (HTTP). FastAPI processes per the existing routing
rules (passthrough or translate), and the response streams back
through mitmproxy to the client.

Hosts not on the allowlist hit `tls_clienthello`, get
`ignore_connection = True`, and the original CONNECT flows through
verbatim; mitmproxy never sees the inner bytes.

## Setup

### 1. Install mitmproxy's CA in your OS trust store

mitmproxy generates a self-signed CA at first run. Trigger it once
(any way) so the cert exists:

```bash
mitmdump --listen-port 8080
# Ctrl-C after a second; we just need the CA generated.
```

The cert lands at `~/.mitmproxy/mitmproxy-ca-cert.pem`. Trust it:

- **macOS**: `security add-trusted-cert -d -r trustRoot -k ~/Library/Keychains/login.keychain ~/.mitmproxy/mitmproxy-ca-cert.pem`
- **Linux** (Debian/Ubuntu): copy the PEM to `/usr/local/share/ca-certificates/mitmproxy.crt` (renamed `.crt`) and run `sudo update-ca-certificates`
- **Windows**: import `~/.mitmproxy/mitmproxy-ca-cert.cer` into "Trusted Root Certification Authorities" via `certmgr.msc` or
  `certutil -addstore -f "ROOT" %USERPROFILE%\.mitmproxy\mitmproxy-ca-cert.cer`

The full reference (with screenshots) is at
<https://docs.mitmproxy.org/stable/concepts-certificates/>. This is a
one-time, per-machine step. Without it, every intercepted request
fails with a TLS error.

### 2. Enable ingress in `magos.yaml`

```yaml
ingress:
  http:
    host: 127.0.0.1
    port: 6246
  mitm:
    enabled: true
    host: 127.0.0.1
    port: 6247
    intercept_hosts:
      - api.anthropic.com
```

`ingress.http.host` / `ingress.http.port` are the FastAPI bind.
`MAGOS_HOST` / `MAGOS_PORT` env vars override yaml when set; CLI
`--host` / `--port` flags override env.

`intercept_hosts` lists the hosts (and their subdomains) magos should
MITM. Anything else passes through un-touched.

### 3. Run magos

```bash
uv run magos serve
```

Both listeners come up. Look for:

```
ingress.started listen=127.0.0.1:6247 target=127.0.0.1:6246 intercept_hosts=['api.anthropic.com']
```

### 4. Point Claude Code at it

```bash
HTTPS_PROXY=http://127.0.0.1:6247 claude --print "hi"
```

Claude Code keeps using `api.anthropic.com` as the base URL and its
behavior is unchanged. magos's structlog will show a `route.matched`
event for each request: the existing Anthropic passthrough rule
fires as if the client had hit FastAPI directly.

## Loop hazard

**Do not set `HTTPS_PROXY` system-wide while magos is running.** If
the magos process inherits `HTTPS_PROXY=http://127.0.0.1:6247`, its
own outbound httpx (passthrough/translate calls to
`api.anthropic.com`) will re-enter mitmproxy, and you'll see infinite
loops or stalls.

Two safe patterns:

1. Set `HTTPS_PROXY` in the **client's** environment only (e.g.
   `HTTPS_PROXY=… claude …` or `direnv` in the client's working
   directory). Don't export it at the shell level.
2. If you must set it globally, also set
   `NO_PROXY=localhost,127.0.0.1` so magos's loopback calls bypass
   mitmproxy.

The addon includes a defensive guard: a request that arrives already
addressed at the FastAPI target is passed through unchanged rather
than re-rewritten. That keeps the loop visible (you'll see one extra
hop) instead of infinite, but it doesn't fix the underlying loop.

## Troubleshooting

- **TLS / certificate errors at the client**: the CA isn't trusted yet.
  Repeat step 1.
- **Intercepted request never reaches FastAPI**: check
  `intercept_hosts` includes the host the client is using. SNI
  matching is exact-or-subdomain (e.g. `api.anthropic.com` matches
  `eu.api.anthropic.com`).
- **`Address already in use` on :6247**: another mitmdump or service
  has the port. Change `ingress.mitm.port` in yaml (or set
  `MAGOS_MITM_PORT`, or pass `--mitm-port`).
- **Overriding mitm settings via env**: every `ingress.mitm.*` yaml
  key has a matching `MAGOS_MITM_*` env var (resolved in
  `serve.resolve_mitm`):
  - `MAGOS_MITM_ENABLED` (`true`/`false`) — toggle the listener
  - `MAGOS_MITM_HOST` — bind host
  - `MAGOS_MITM_PORT` — bind port
  - `MAGOS_MITM_INTERCEPT_HOSTS` — comma-separated host list
    (e.g. `api.anthropic.com,api.openai.com`)
  Env wins over yaml; CLI flags (`--mitm-port`) win over env.
- **Clients that don't honor `HTTPS_PROXY`**: some tools have their
  own proxy logic. Check the tool's docs; the `HTTP_PROXY` /
  `HTTPS_PROXY` env vars are the de-facto standard but not universal.

## What stays unchanged

- All routing rules, rewrites, and dispatch behavior: same as a
  client hitting FastAPI directly.
- Anthropic prompt-cache hashes: mitmproxy streams the body through
  byte-exact, and magos's passthrough mode forwards bytes verbatim
  upstream.
- OAuth tokens (`sk-ant-oat...`): auth-header injection still runs
  on the FastAPI side; the ingress addon doesn't touch headers.
- Egress observability: `MagosObserverAddon` is loaded alongside the
  ingress addon, so outbound LLM provider traffic that does pass
  through mitmproxy (e.g. via translate-mode `litellm`'s default
  httpx) gets the same `egress.request` / `egress.response` events as
  before. Note this only fires when magos's outbound is configured to
  use the local mitmproxy as a proxy, which we explicitly recommend
  against (see "Loop hazard").

## Disabling

Set `ingress.mitm.enabled: false` (or remove the `mitm:` sub-block
entirely). FastAPI keeps running on its bind address; the mitm task
isn't started; nothing else changes.
