"""Round-trip tests for the reverse translation direction.

- request_openai_to_anthropic   takes openai_request.json   -> anthropic_request.json
- response_anthropic_to_openai  takes anthropic_response.json -> openai_response.json

Some cases do not round-trip cleanly because both APIs admit multiple
equivalent shapes for the same logical content. ``SKIP_REVERSE_REQUEST``
documents those, with the reason. Add to it deliberately, not as a workaround.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from magos.translation import (
    request_openai_to_anthropic,
    response_anthropic_to_openai,
)

FIXTURES_ROOT = Path(__file__).parent / "fixtures" / "translation"

# content_blocks: OpenAI uses a flat string ("Part one. Part two."); Anthropic
# request explicitly uses two text blocks. The reverse translator collapses
# single-text content to a string, which is the more idiomatic shape but does
# not preserve the original block split.
SKIP_REVERSE_REQUEST = {"content_blocks"}


def _load(case_dir: Path, name: str) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((case_dir / name).read_text(encoding="utf-8")))


def _case_dirs() -> list[Path]:
    if not FIXTURES_ROOT.is_dir():
        return []
    return sorted(p for p in FIXTURES_ROOT.iterdir() if p.is_dir())


CASES = _case_dirs()
CASE_IDS = [p.name for p in CASES]


@pytest.mark.unit
@pytest.mark.parametrize("case_dir", CASES, ids=CASE_IDS)
def test_request_openai_to_anthropic(case_dir: Path) -> None:
    if case_dir.name in SKIP_REVERSE_REQUEST:
        pytest.skip(f"{case_dir.name} does not round-trip cleanly (see module docstring)")
    openai_req = _load(case_dir, "openai_request.json")
    expected = _load(case_dir, "anthropic_request.json")

    actual = request_openai_to_anthropic(openai_req)

    # Anthropic accepts request-level fields the proxy does not synthesize
    # (top_k, metadata.user_id when not in OpenAI). Drop fields the reverse
    # direction can't reconstruct.
    expected_filtered = {k: v for k, v in expected.items() if k != "top_k"}
    assert actual == expected_filtered


@pytest.mark.unit
@pytest.mark.parametrize("case_dir", CASES, ids=CASE_IDS)
def test_response_anthropic_to_openai(case_dir: Path) -> None:
    anthropic_resp = _load(case_dir, "anthropic_response.json")
    expected = _load(case_dir, "openai_response.json")

    actual = response_anthropic_to_openai(
        anthropic_resp,
        response_id=expected["id"],
        created=expected["created"],
    )
    assert actual == expected
