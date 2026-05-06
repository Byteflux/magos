# Observability

OTel metrics (`magos.registry.*`) emitted by the refresher:

| Metric | Type | Notes |
|--------|------|-------|
| `refresh.total{provider, status}` | counter | `attempt`, `success`, `failure` |
| `refresh.failures{provider, error_type}` | counter | |
| `refresh.duration` | histogram | seconds, per provider |
| `models.total{provider}` | observable gauge | count of registry entries per provider that have not been pruned (includes entries soft-deleted within the deprecation grace window) |
| `models.added{provider}`, `models.deprecated{provider}`, `models.pruned{provider}` | counters | |

Set `MAGOS_METRICS_ENABLED=1` to install the OTel Prometheus exporter
at startup and mount the `GET /metrics` endpoint. Without the env var,
the meters bind to OTel's no-op default and `/metrics` is not served.

structlog events:

| Event | Level | Notes |
|-------|-------|-------|
| `registry.refresh.attempt` | debug | per refresh start |
| `registry.refresh.success` | info | includes added/deprecated/pruned counts |
| `registry.refresh.failure` | warning | includes error and error_type |
| `registry.auto_route` | debug | when auto-routing picks a provider |
