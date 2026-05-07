"""Factory for ``RequestService``. Shared by every ingress surface."""

from __future__ import annotations

from magos.registry.refresher import Refresher
from magos.registry.schema import RegistryYaml
from magos.routing import RoutingConfig

from .request import RequestService


def build_request_service(
    cfg: RoutingConfig,
    refresher: Refresher | None,
    registry_cfg: RegistryYaml,
) -> RequestService:
    """Construct the ``RequestService`` from long-lived collaborators.

    Phases C1, C2, C3 will expand this factory to wire injected
    ``Router``, ``Gateway``, and ``Compressor`` collaborators.
    """
    return RequestService(cfg=cfg, refresher=refresher, registry_cfg=registry_cfg)
