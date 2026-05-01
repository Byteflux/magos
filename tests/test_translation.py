"""Contract tests pinning Anthropic <-> OpenAI translation against golden fixtures.

Each case directory under tests/fixtures/translation/ holds four JSON files:

    anthropic_request.json   -> input to request_anthropic_to_openai
    openai_request.json      -> expected output of request_anthropic_to_openai
    openai_response.json     -> input to response_openai_to_anthropic
    anthropic_response.json  -> expected output of response_openai_to_anthropic

Tests are intentionally RED until magos.translation exists (step 2 of the slice).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from magos.translation import (
    request_anthropic_to_openai,
    response_openai_to_anthropic,
)

FIXTURES_ROOT = Path(__file__).parent / "fixtures" / "translation"


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
def test_request_anthropic_to_openai(case_dir: Path) -> None:
    anthropic_req = _load(case_dir, "anthropic_request.json")
    expected = _load(case_dir, "openai_request.json")
    assert request_anthropic_to_openai(anthropic_req) == expected


@pytest.mark.unit
@pytest.mark.parametrize("case_dir", CASES, ids=CASE_IDS)
def test_response_openai_to_anthropic(case_dir: Path) -> None:
    openai_resp = _load(case_dir, "openai_response.json")
    expected = _load(case_dir, "anthropic_response.json")
    actual = response_openai_to_anthropic(openai_resp)
    expected_no_id = {k: v for k, v in expected.items() if k != "id"}
    actual_no_id = {k: v for k, v in actual.items() if k != "id"}
    assert actual_no_id == expected_no_id
    assert isinstance(actual.get("id"), str) and actual["id"].startswith("msg_")
