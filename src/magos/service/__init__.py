"""Application Service Layer — see ``docs/architecture/migration.md``.

The service is the boundary between ingress surfaces (FastAPI, mitmproxy)
and the domain logic (routing + dispatch). One ``RequestService``
instance is constructed per app and shared by all in-flight requests.
"""

from __future__ import annotations

from .build import build_request_service
from .request import RequestService, RoutedResponse

__all__ = ["RequestService", "RoutedResponse", "build_request_service"]
