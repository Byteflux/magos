"""Shared CLI helpers used across subapps.

The ``magos models`` subapp reads registry state via two paths: the
running server's admin endpoints, or the on-disk ``models.json``. The
helpers here centralise that resolution so future subapps that want the
same fallback semantics can reuse it without duplicating the env-over-yaml
bind layering or the source-attribution string.
"""

from __future__ import annotations

import json
from enum import StrEnum

import typer

from magos.cli.admin_client import AdminClient, AdminClientError
from magos.config.loader import load_full_config, resolve_models_path
from magos.config.settings import MagosSettings
from magos.registry.state import RegistryState
from magos.registry.store import deserialize, load


class ListFormat(StrEnum):
    """Output format selector for read commands."""

    text = "text"
    json = "json"


def admin_client(settings: MagosSettings) -> AdminClient:
    """Build an admin client targeting the local server's bind address.

    Mirrors the env-over-yaml layering :func:`magos.serve.resolve_bind`
    uses so the CLI hits the same listener the orchestrator opens.
    """
    from magos.serve import resolve_bind  # noqa: PLC0415  - keeps cli import tree light

    cfg = load_full_config(settings.config_path)
    resolved_host, resolved_port = resolve_bind(settings, cfg.ingress.http)
    # Bind addresses like 0.0.0.0 / :: aren't valid HTTP hosts; resolve to
    # loopback so the CLI talks to the local instance.
    bind_all = {"0.0.0.0", "::"}  # noqa: S104  - not binding, just comparing
    host = "127.0.0.1" if resolved_host in bind_all else resolved_host
    return AdminClient(f"http://{host}:{resolved_port}")


def load_state_from_disk(settings: MagosSettings) -> RegistryState:
    cfg = load_full_config(settings.config_path)
    return load(resolve_models_path(cfg.registry, override=settings.models_path))


def load_state(settings: MagosSettings, *, prefer_disk: bool) -> tuple[RegistryState, str]:
    """Return ``(state, source)`` where ``source`` is ``'server'`` or ``'disk'``."""
    if prefer_disk:
        return load_state_from_disk(settings), "disk"
    client = admin_client(settings)
    try:
        raw = client.get_registry()
    except AdminClientError as exc:
        typer.echo(f"server returned an error: {exc}")
        raise typer.Exit(2) from exc
    if raw is not None:
        return deserialize(raw), "server"
    typer.echo("server unreachable, falling back to disk")
    return load_state_from_disk(settings), "disk"


def print_list(state: RegistryState, source: str, *, fmt: ListFormat) -> None:
    """Print a registry summary in either ``text`` or ``json`` form."""
    entries = sorted(state.entries.values(), key=lambda e: e.namespaced_id)
    if fmt is ListFormat.json:
        payload = [
            {
                "id": e.namespaced_id,
                "litellm_id": e.litellm_id,
                "context_size": e.context_size,
                "deprecated": e.is_deprecated,
            }
            for e in entries
        ]
        typer.echo(json.dumps({"source": source, "entries": payload}, indent=2))
        return
    typer.echo(f"# source: {source}")
    if not entries:
        typer.echo("(no entries)")
        return
    for entry in entries:
        marker = " [deprecated]" if entry.is_deprecated else ""
        ctx = f" ctx={entry.context_size}" if entry.context_size else ""
        typer.echo(f"{entry.namespaced_id}{ctx}{marker}")
