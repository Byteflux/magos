"""Ingress: how requests enter magos.

Two entry points share the same routing engine:

- ``magos.ingress.http`` (currently lives in ``magos.server``) — FastAPI
  endpoints clients hit directly.
- ``magos.ingress.mitm`` — optional in-process mitmproxy listener that
  terminates TLS for ``HTTPS_PROXY`` clients (e.g. Claude Code, where
  ``ANTHROPIC_BASE_URL`` would change behavior) and rewrites requests
  to FastAPI loopback.

Routing decisions and dispatch live in :mod:`magos.routing` and
:mod:`magos.egress` respectively, regardless of how the request entered.
"""
