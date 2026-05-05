# Overview

## Why

Without the registry, every routing rule had to enumerate models by
regex or literal. Onboarding a new provider meant editing yaml. The
registry inverts that: providers describe themselves over their own
discovery API, magos merges with operator overrides and LiteLLM's
bundled metadata, and routing falls back to the registry when no
explicit rule applies.

## Lifecycle

```
boot
 ├── load models.json from disk      (regenerable cache, no schema versioning)
 ├── for each provider with no entries:
 │     run discovery once with tight timeout (10s, 1 attempt)
 │     populate state, persist
 └── start per-provider background loop
       ├── sleep refresh_interval (default 2h, per-provider override)
       ├── refresh with patient timeout (30s, 3 attempts, exponential backoff)
       ├── apply deprecation state machine
       └── atomic state swap, persist to models.json
```

Failure modes:

- **Boot discovery fails**: that provider boots empty; other providers
  unaffected. The background loop will retry on its normal cadence.
- **Background refresh fails**: prior state preserved (atomic). Failure
  metric increments; logs include the error type. Next tick tries again.
- **Provider drops a model**: the entry is marked `deprecated_at = now`
  and continues serving. If absent for 3 days (configurable), the entry
  is hard-deleted on the next refresh that includes that provider.
- **Model reappears mid-grace**: the deprecation mark is cleared.
- **Corrupt models.json**: file is treated as missing; live discovery
  rebuilds. No schema versioning by design.
