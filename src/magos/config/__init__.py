"""Configuration: process-level env settings, yaml schema, and loader.

- :mod:`magos.config.settings` — :class:`MagosSettings` (pydantic-settings,
  env-driven; ``MAGOS_*`` overrides). Anchors :func:`magos_home`.
- :mod:`magos.config.schema` — :class:`MagosServerConfig` /
  :class:`IngressConfig` for the ``server:`` block in ``magos.yaml``.
- :mod:`magos.config.loader` — :func:`load_full_config` parses routing,
  registry, and server blocks into a single :class:`MagosConfig`.
"""
