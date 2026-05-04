"""FastAPI ingress: the default entry point for client traffic.

:func:`create_app` is the canonical builder. Submodules:

- :mod:`magos.ingress.http.app` — :func:`create_app`
- :mod:`magos.ingress.http.lifespan` — startup/shutdown coordination
- :mod:`magos.ingress.http.handlers` — DI seams + 7 endpoint handlers
- :mod:`magos.ingress.http.run` — shared dispatch flow
- :mod:`magos.ingress.http.headers` — inbound header filter
- :mod:`magos.ingress.http.admin` — ``/admin/registry/*`` endpoints
"""

from __future__ import annotations

from magos.ingress.http.app import create_app

__all__ = ["create_app"]
