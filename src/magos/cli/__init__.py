"""Operator-facing CLI for magos.

Subcommand layout::

    magos serve            # run the FastAPI server (and optional mitm)
    magos models list      # show in-memory state from running server
    magos models show <id>
    magos models refresh [--provider X]
    magos models prune
    magos models discover --provider X --dry-run

Read commands (``list`` / ``show``) try the running server's admin
endpoints first, then fall back to the on-disk ``models.json`` if the
server isn't reachable. Mutating commands (``refresh`` / ``prune``)
require the server to be running.
"""
