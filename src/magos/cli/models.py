"""``magos models`` Typer subapp: inspect and manage the model registry.

Read commands (``list`` / ``show``) try the running server's admin
endpoints first and fall back to the on-disk ``models.json`` if the
server isn't reachable. Mutating commands (``refresh`` / ``prune``)
require the server. ``discover`` is the odd one out — it bypasses the
server entirely and queries the discovery adapter directly, useful for
debugging a provider before wiring it into ``magos.yaml``.

Per-state-loading and print helpers live in :mod:`magos.cli._helpers`
so future subapps can reuse them without re-implementing the
env-over-yaml bind layering.
"""

from __future__ import annotations

import asyncio
import json
from typing import Annotated

import httpx
import typer

from magos.cli import _helpers
from magos.cli._helpers import ListFormat
from magos.cli.admin_client import AdminClientError
from magos.config.loader import load_full_config
from magos.config.settings import MagosSettings
from magos.registry.discovery import adapter_for
from magos.registry.discovery.base import DiscoveryError

models_app = typer.Typer(
    name="models",
    help="Inspect and manage the model registry.",
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@models_app.command("list")
def models_list(
    from_disk: Annotated[
        bool, typer.Option("--from-disk", help="Bypass the server and read models.json directly.")
    ] = False,
    output_format: Annotated[
        ListFormat, typer.Option("--format", help="Output format.")
    ] = ListFormat.text,
) -> None:
    """Show registry entries (server-state by default, --from-disk to bypass)."""
    settings = MagosSettings()
    state, source = _helpers.load_state(settings, prefer_disk=from_disk)
    _helpers.print_list(state, source, fmt=output_format)


@models_app.command("show")
def models_show(
    model_id: Annotated[
        str, typer.Argument(help="Namespaced id, e.g. openrouter/anthropic/claude-sonnet-4-6.")
    ],
    from_disk: Annotated[bool, typer.Option("--from-disk")] = False,
) -> None:
    """Show one entry by namespaced id."""
    settings = MagosSettings()
    state, source = _helpers.load_state(settings, prefer_disk=from_disk)
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
        "cache_read_cost": entry.cache_read_cost,
        "cache_write_cost": entry.cache_write_cost,
        "input_modalities": list(entry.input_modalities),
        "output_modalities": list(entry.output_modalities),
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
    client = _helpers.admin_client(settings)
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
    client = _helpers.admin_client(settings)
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
