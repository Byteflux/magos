"""Pytest fixtures and bootstrap.

Loads ``.env`` into ``os.environ`` only when end-to-end tests are enabled, so
unit and integration runs stay free of real provider credentials. LiteLLM and
other libraries read keys directly from the process environment, not from
pydantic-settings, hence the explicit population here.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

if os.environ.get("MAGOS_E2E") == "1":
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)
