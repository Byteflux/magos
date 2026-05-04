"""Smoke test for ``python -m magos`` entrypoint shim."""

from __future__ import annotations


def test_main_module_re_exports_cli_main() -> None:
    """``python -m magos`` resolves to the CLI's ``main()``.

    The shim is two lines (``from magos.cli.app import main`` plus the
    ``if __name__ == "__main__":`` guard); this test imports it so the
    re-export line is covered, and asserts the symbol is callable.
    """
    import magos.__main__ as entry  # noqa: PLC0415
    from magos.cli.app import main  # noqa: PLC0415

    assert entry.main is main  # type: ignore[attr-defined]
    assert callable(entry.main)  # type: ignore[attr-defined]
