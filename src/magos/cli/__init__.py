"""Operator-facing CLI for magos.

Subcommand layout::

    python -m magos                  # serve (default)
    python -m magos serve            # explicit form
    python -m magos models list      # show in-memory state from running server
    python -m magos models show <id>
    python -m magos models refresh [--provider X]
    python -m magos models prune
    python -m magos models discover --provider X --dry-run

Read commands (``list`` / ``show``) try the running server's admin
endpoints first, then fall back to the on-disk ``models.json`` if the
server isn't reachable. Mutating commands (``refresh`` / ``prune``)
require the server to be running.
"""
