"""Model registry: discovery, persistence, and lookup of provider-served models.

Public surface re-exports the core data shapes; submodules own behavior.
"""

from __future__ import annotations

from magos.registry.models import ModelEntry, RegistryState

__all__ = ["ModelEntry", "RegistryState"]
