"""``magos models`` subcommands."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import TextIO

import httpx

from magos.cli.admin_client import AdminClient, AdminClientError
from magos.config import MagosSettings
from magos.config_loader import load_full_config, resolve_models_path
from magos.registry.discovery import adapter_for
from magos.registry.discovery.base import DiscoveryError
from magos.registry.models import RegistryState
from magos.registry.refresher import Refresher
from magos.registry.store import deserialize, load


def _admin_client(settings: MagosSettings) -> AdminClient:
    """Build an admin client that targets the local server's bind address."""
    # Bind addresses like 0.0.0.0 / :: aren't valid HTTP hosts; resolve to
    # loopback so the CLI talks to the local instance.
    bind_all = {"0.0.0.0", "::"}  # noqa: S104  - not binding, just comparing
    host = "127.0.0.1" if settings.host in bind_all else settings.host
    return AdminClient(f"http://{host}:{settings.port}")


def _load_state_from_disk(settings: MagosSettings) -> RegistryState:
    cfg = load_full_config(settings.config_path)
    return load(resolve_models_path(settings.config_path, cfg.registry))


def _load_state(
    settings: MagosSettings, *, prefer_disk: bool, out: TextIO
) -> tuple[RegistryState, str]:
    """Return (state, source) where source is 'server' or 'disk'."""
    if prefer_disk:
        return _load_state_from_disk(settings), "disk"
    client = _admin_client(settings)
    try:
        raw = client.get_registry()
    except AdminClientError as exc:
        print(f"server returned an error: {exc}", file=out)
        raise SystemExit(2) from exc
    if raw is not None:
        return deserialize(raw), "server"
    print("server unreachable, falling back to disk", file=out)
    return _load_state_from_disk(settings), "disk"


def _print_list(state: RegistryState, source: str, *, fmt: str, out: TextIO) -> None:
    entries = sorted(state.entries.values(), key=lambda e: e.namespaced_id)
    if fmt == "json":
        payload = [
            {
                "id": e.namespaced_id,
                "litellm_id": e.litellm_id,
                "context_size": e.context_size,
                "deprecated": e.is_deprecated,
            }
            for e in entries
        ]
        print(json.dumps({"source": source, "entries": payload}, indent=2), file=out)
        return
    print(f"# source: {source}", file=out)
    if not entries:
        print("(no entries)", file=out)
        return
    for entry in entries:
        marker = " [deprecated]" if entry.is_deprecated else ""
        ctx = f" ctx={entry.context_size}" if entry.context_size else ""
        print(f"{entry.namespaced_id}{ctx}{marker}", file=out)


def _print_show(state: RegistryState, source: str, model_id: str, *, out: TextIO) -> int:
    entry = state.get(model_id)
    if entry is None:
        print(f"not found in {source}: {model_id}", file=out)
        return 1
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
    print(json.dumps(payload, indent=2), file=out)
    return 0


def _cmd_list(args: argparse.Namespace, settings: MagosSettings, out: TextIO) -> int:
    state, source = _load_state(settings, prefer_disk=args.from_disk, out=out)
    _print_list(state, source, fmt=args.format, out=out)
    return 0


def _cmd_show(args: argparse.Namespace, settings: MagosSettings, out: TextIO) -> int:
    state, source = _load_state(settings, prefer_disk=args.from_disk, out=out)
    return _print_show(state, source, args.id, out=out)


def _cmd_refresh(args: argparse.Namespace, settings: MagosSettings, out: TextIO) -> int:
    client = _admin_client(settings)
    try:
        result = client.post_refresh(provider=args.provider)
    except AdminClientError as exc:
        print(f"refresh failed: {exc}", file=out)
        return 2
    print(json.dumps(result, indent=2), file=out)
    return 0 if not result.get("failed") else 1


def _cmd_prune(args: argparse.Namespace, settings: MagosSettings, out: TextIO) -> int:
    client = _admin_client(settings)
    try:
        result = client.post_prune()
    except AdminClientError as exc:
        print(f"prune failed: {exc}", file=out)
        return 2
    print(json.dumps(result, indent=2), file=out)
    return 0


def _cmd_discover(args: argparse.Namespace, settings: MagosSettings, out: TextIO) -> int:
    """Standalone discovery against one provider (no server, no writes)."""
    cfg = load_full_config(settings.config_path)
    if args.provider not in cfg.registry.providers:
        print(f"unknown provider: {args.provider}", file=out)
        return 2
    provider_cfg = cfg.registry.providers[args.provider]
    adapter = adapter_for(provider_cfg)

    async def run() -> int:
        timeout = cfg.registry.registry.discovery_timeout_seconds
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                result = await adapter.discover(args.provider, provider_cfg, client)
            except DiscoveryError as exc:
                print(f"discovery failed: {exc}", file=out)
                return 2
        if not args.dry_run:
            print(
                "(non-dry-run discovery would normally update models.json; "
                "use --dry-run for the read-only path)",
                file=out,
            )
        for entry in result.models:
            print(entry.raw_id, file=out)
        return 0

    return asyncio.run(run())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="magos models")
    sub = parser.add_subparsers(dest="verb", required=True)

    # list
    p_list = sub.add_parser("list", help="show registry entries")
    p_list.add_argument("--from-disk", action="store_true", help="bypass server, read models.json")
    p_list.add_argument("--format", choices=("text", "json"), default="text")
    p_list.set_defaults(func=_cmd_list)

    # show
    p_show = sub.add_parser("show", help="show one entry by namespaced id")
    p_show.add_argument("id", help="namespaced id, e.g. openrouter/anthropic/claude-sonnet-4-6")
    p_show.add_argument("--from-disk", action="store_true")
    p_show.set_defaults(func=_cmd_show)

    # refresh
    p_refresh = sub.add_parser("refresh", help="trigger a refresh via the running server")
    p_refresh.add_argument("--provider", help="scope to one provider (default: all)")
    p_refresh.set_defaults(func=_cmd_refresh)

    # prune
    p_prune = sub.add_parser("prune", help="trigger a deprecation sweep via refresh")
    p_prune.set_defaults(func=_cmd_prune)

    # discover
    p_discover = sub.add_parser("discover", help="standalone read-only discovery")
    p_discover.add_argument("--provider", required=True)
    p_discover.add_argument("--dry-run", action="store_true", default=True)
    p_discover.set_defaults(func=_cmd_discover)

    return parser


def main(argv: list[str] | None = None, *, out: TextIO | None = None) -> int:
    """Entrypoint for ``python -m magos models …``."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    settings = MagosSettings()
    target = out if out is not None else sys.stdout
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help(target)
        return 1
    return int(func(args, settings, target))


# Suppress unused-import warning; ``Refresher`` is referenced by future
# expansion of the discover subcommand to the write path.
_ = Refresher
