# Pipeline

## Setup

```bash
mkdir -p ~/.magos
cp magos.example.yaml ~/.magos/magos.yaml
# edit ~/.magos/magos.yaml
magos                                    # picks up the default
magos --config /etc/magos.yaml           # CLI override
MAGOS_CONFIG_PATH=/etc/magos.yaml magos  # env override
```

Config path resolution (highest wins): `--config` flag, then
`MAGOS_CONFIG_PATH`, then `$MAGOS_HOME/magos.yaml` (which falls back
to `~/.magos/magos.yaml` when `MAGOS_HOME` is unset).

`MAGOS_HOME` is the magos data directory: it anchors the default
location of both `magos.yaml` and the registry's `models.json`. Set
it once (e.g. `MAGOS_HOME=/srv/magos`) and both files default into
the same directory without editing the yaml.

## Pipeline

```
inbound request
  -> pre_transforms        (global, applied unconditionally, top-to-bottom)
  -> match                 (against transformed request)
  -> post_transforms       (per matched rule, top-to-bottom)
  -> dispatch via action
```

Rules are evaluated **first-match-wins**. If you want a fallback, declare
it last.
