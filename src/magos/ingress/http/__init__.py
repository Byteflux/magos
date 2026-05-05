"""FastAPI ingress: the default entry point for client traffic."""

from __future__ import annotations

from magos.ingress.http.app import create_app

__all__ = ["create_app"]
