"""End-to-end pipeline tests for proxy_anthropic_messages.

Drives the full pipeline against every golden case with an injected fake
``completion`` callable. Verifies both directions: the OpenAI request the
upstream would have seen, and the Anthropic response the client receives.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest

from magos.proxy import proxy_anthropic_messages

FIXTURES_ROOT = Path(__file__).parent / "fixtures" / "translation"


def _load(case_dir: Path, name: str) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((case_dir / name).read_text(encoding="utf-8")))


def _case_dirs() -> list[Path]:
    if not FIXTURES_ROOT.is_dir():
        return []
    return sorted(p for p in FIXTURES_ROOT.iterdir() if p.is_dir())


CASES = _case_dirs()
CASE_IDS = [p.name for p in CASES]


@pytest.mark.integration
@pytest.mark.parametrize("case_dir", CASES, ids=CASE_IDS)
def test_proxy_pipeline_round_trip(case_dir: Path) -> None:
    anthropic_request = _load(case_dir, "anthropic_request.json")
    expected_openai_request = _load(case_dir, "openai_request.json")
    openai_response = _load(case_dir, "openai_response.json")
    expected_anthropic_response = _load(case_dir, "anthropic_response.json")

    received: dict[str, Any] = {}

    async def fake_completion(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return openai_response

    dispatch_model = f"anthropic/{expected_openai_request['model']}"
    actual = asyncio.run(
        proxy_anthropic_messages(
            anthropic_request,
            dispatch_model=dispatch_model,
            completion=fake_completion,
        )
    )

    # The router supplies dispatch_model; proxy uses it verbatim instead
    # of inferring from a bare model name.
    expected_normalised = {**expected_openai_request, "model": dispatch_model}
    assert received == expected_normalised

    expected_no_id = {k: v for k, v in expected_anthropic_response.items() if k != "id"}
    actual_no_id = {k: v for k, v in actual.items() if k != "id"}
    assert actual_no_id == expected_no_id
    assert isinstance(actual.get("id"), str) and actual["id"].startswith("msg_")


@pytest.mark.integration
def test_proxy_pipeline_accepts_pydantic_like_response() -> None:
    anthropic_request = _load(CASES[0], "anthropic_request.json")
    openai_response = _load(CASES[0], "openai_response.json")

    class _PydanticLike:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def model_dump(self) -> dict[str, Any]:
            return self._payload

    async def fake_completion(**kwargs: Any) -> _PydanticLike:
        return _PydanticLike(openai_response)

    result = asyncio.run(
        proxy_anthropic_messages(
            anthropic_request,
            dispatch_model="anthropic/dummy",
            completion=fake_completion,
        )
    )
    assert result["type"] == "message"
    assert result["role"] == "assistant"
