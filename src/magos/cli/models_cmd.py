"""``magos models`` Typer subapp."""

from __future__ import annotations

import asyncio
import json
from enum import StrEnum
from typing import Annotated

import httpx
import typer

from magos.cli.admin_client import AdminClient, AdminClientError
from magos.config import MagosSettings
from magos.config_loader import load_full_config, resolve_models_path
from magos.registry.discovery import adapter_for
from magos.registry.discovery.base import DiscoveryError
from magos.registry.models import RegistryState
from magos.registry.store import deserialize, load


class _ListFormat(StrEnum):
    text = "text"
    json = "json"


models_app = typer.Typer(
    name="models",
    help="Inspect and manage the model registry.",
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _admin_client(settings: MagosSettings) -> AdminClient:
    """Build an admin client targeting the local server's bind address.

    Mirrors the env-over-yaml layering :func:`magos.serve.resolve_bind`
    uses so the CLI hits the same listener the orchestrator opens.
    """
    from magos.serve import resolve_bind  # noqa: PLC0415  - keeps cli import tree light

    cfg = load_full_config(settings.config_path)
    resolved_host, resolved_port = resolve_bind(settings, cfg.server)
    # Bind addresses like 0.0.0.0 / :: aren't valid HTTP hosts; resolve to
    # loopback so the CLI talks to the local instance.
    bind_all = {"0.0.0.0", "::"}  # noqa: S104  - not binding, just comparing
    host = "127.0.0.1" if resolved_host in bind_all else resolved_host
    return AdminClient(f"http://{host}:{resolved_port}")


def _load_state_from_disk(settings: MagosSettings) -> RegistryState:
    cfg = load_full_config(settings.config_path)
    return load(resolve_models_path(cfg.registry, override=settings.models_path))


def _load_state(settings: MagosSettings, *, prefer_disk: bool) -> tuple[RegistryState, str]:
    """Return ``(state, source)`` where ``source`` is ``'server'`` or ``'disk'``."""
    if prefer_disk:
        return _load_state_from_disk(settings), "disk"
    client = _admin_client(settings)
    try:
        raw = client.get_registry()
    except AdminClientError as exc:
        typer.echo(f"server returned an error: {exc}")
        raise typer.Exit(2) from exc
    if raw is not None:
        return deserialize(raw), "server"
    typer.echo("server unreachable, falling back to disk")
    return _load_state_from_disk(settings), "disk"


def _print_list(state: RegistryState, source: str, *, fmt: _ListFormat) -> None:
    entries = sorted(state.entries.values(), key=lambda e: e.namespaced_id)
    if fmt is _ListFormat.json:
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


@models_app.command("list")
def models_list(
    from_disk: Annotated[
        bool, typer.Option("--from-disk", help="Bypass the server and read models.json directly.")
    ] = False,
    output_format: Annotated[
        _ListFormat, typer.Option("--format", help="Output format.")
    ] = _ListFormat.text,
) -> None:
    """Show registry entries (server-state by default, --from-disk to bypass)."""
    settings = MagosSettings()
    state, source = _load_state(settings, prefer_disk=from_disk)
    _print_list(state, source, fmt=output_format)


@models_app.command("show")
def models_show(
    model_id: Annotated[
        str, typer.Argument(help="Namespaced id, e.g. openrouter/anthropic/claude-sonnet-4-6.")
    ],
    from_disk: Annotated[bool, typer.Option("--from-disk")] = False,
) -> None:
    """Show one entry by namespaced id."""
    settings = MagosSettings()
    state, source = _load_state(settings, prefer_disk=from_disk)
    entry = state.get(model_id)
    if entry is None:
        typer.echo(f"not found in {source}: {model_id}")
        raise typer.Exit(1)
    payload = {
        "source": source,
        "provider": entry.provider,
        "raw_id": entry.raw_id,
        "namespaced_id": entry.namespaced_id,
        "litellm_id": entry.litellm_id,
        "context_size": entry.context_size,
        "max_output": entry.max_output,
        "input_cost": entry.input_cost,
        "output_cost": entry.output_cost,
        "modalities": list(entry.modalities),
        "deprecated_at": entry.deprecated_at.isoformat() if entry.deprecated_at else None,
        "sources": list(entry.sources),
    }
    typer.echo(json.dumps(payload, indent=2))


@models_app.command("refresh")
def models_refresh(
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Scope to one provider (default: all)."),
    ] = None,
) -> None:
    """Trigger a refresh via the running server."""
    settings = MagosSettings()
    client = _admin_client(settings)
    try:
        result = client.post_refresh(provider=provider)
    except AdminClientError as exc:
        typer.echo(f"refresh failed: {exc}")
        raise typer.Exit(2) from exc
    typer.echo(json.dumps(result, indent=2))
    if result.get("failed"):
        raise typer.Exit(1)


@models_app.command("prune")
def models_prune() -> None:
    """Trigger a deprecation sweep via refresh."""
    settings = MagosSettings()
    client = _admin_client(settings)
    try:
        result = client.post_prune()
    except AdminClientError as exc:
        typer.echo(f"prune failed: {exc}")
        raise typer.Exit(2) from exc
    typer.echo(json.dumps(result, indent=2))


@models_app.command("discover")
def models_discover(
    provider: Annotated[str, typer.Option("--provider", help="Provider to query.")],
    dry_run: Annotated[
        bool, typer.Option("--dry-run/--no-dry-run", help="Read-only by default.")
    ] = True,
) -> None:
    """Standalone discovery against one provider (no server, no writes)."""
    settings = MagosSettings()
    cfg = load_full_config(settings.config_path)
    if provider not in cfg.registry.providers:
        typer.echo(f"unknown provider: {provider}")
        raise typer.Exit(2)
    provider_cfg = cfg.registry.providers[provider]
    adapter = adapter_for(provider_cfg)

    async def run() -> None:
        timeout = cfg.registry.registry.discovery_timeout_seconds
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                result = await adapter.discover(provider, provider_cfg, client)
            except DiscoveryError as exc:
                typer.echo(f"discovery failed: {exc}")
                raise typer.Exit(2) from exc
        if not dry_run:
            typer.echo(
                "(non-dry-run discovery would normally update models.json; "
                "use --dry-run for the read-only path)"
            )
        for entry in result.models:
            typer.echo(entry.raw_id)

    asyncio.run(run())
