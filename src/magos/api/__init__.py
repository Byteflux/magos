"""FastAPI ingress: the default entry point for client traffic."""

from __future__ import annotations

from magos.api.build import build_api

__all__ = ["build_api"]
