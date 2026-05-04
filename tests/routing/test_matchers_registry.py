"""Tests for the ``model_field`` matcher atom against the registry."""

from __future__ import annotations

import pytest

from magos.registry.state import ModelEntry, RegistryState
from magos.routing.matchers import matches
from magos.routing.request import RoutedRequest
from magos.routing.schema import ModelFieldAtom

from ._helpers import make_registry, make_req


def _request(model: str) -> RoutedRequest:
    return make_req(body={"model": model})


def _registry(entry: ModelEntry) -> RegistryState:
    return make_registry(entry)


def _entry(**fields: object) -> ModelEntry:
    base: dict[str, object] = {
        "provider": "openrouter",
        "raw_id": "anthropic/claude-sonnet-4-6",
        "litellm_id": "openrouter/anthropic/claude-sonnet-4-6",
    }
    base.update(fields)
    return ModelEntry(**base)  # type: ignore[arg-type]


def _atom(field: str, op: str, value: object) -> ModelFieldAtom:
    return ModelFieldAtom.model_validate(
        {"model_field": {"field": field, "op": op, "value": value}}
    )


@pytest.mark.parametrize(
    ("op", "value", "context_size", "expected"),
    [
        ("eq", 200000, 200000, True),
        ("eq", 200000, 100000, False),
        ("gte", 200000, 200000, True),
        ("gte", 200000, 199999, False),
        ("gt", 200000, 200001, True),
        ("gt", 200000, 200000, False),
        ("lt", 200000, 199999, True),
        ("lt", 200000, 200000, False),
        ("lte", 200000, 200000, True),
        ("lte", 200000, 200001, False),
    ],
)
def test_numeric_ops_against_context_size(
    op: str, value: int, context_size: int, expected: bool
) -> None:
    entry = _entry(context_size=context_size)
    request = _request(entry.namespaced_id)
    result = matches(_atom("context_size", op, value), request, registry=_registry(entry))
    assert result is expected


def test_contains_against_input_modalities() -> None:
    entry = _entry(input_modalities=("text", "image"))
    request = _request(entry.namespaced_id)
    assert matches(
        _atom("input_modalities", "contains", "image"), request, registry=_registry(entry)
    )
    assert not matches(
        _atom("input_modalities", "contains", "audio"), request, registry=_registry(entry)
    )


def test_contains_against_output_modalities() -> None:
    entry = _entry(output_modalities=("text", "audio"))
    request = _request(entry.namespaced_id)
    assert matches(
        _atom("output_modalities", "contains", "audio"), request, registry=_registry(entry)
    )


def test_in_against_scalar_field() -> None:
    entry = _entry(context_size=200000)
    request = _request(entry.namespaced_id)
    atom = _atom("context_size", "in", [128000, 200000, 1000000])
    assert matches(atom, request, registry=_registry(entry))


def test_returns_false_when_field_unset() -> None:
    entry = _entry()  # context_size None
    request = _request(entry.namespaced_id)
    assert not matches(_atom("context_size", "gte", 1), request, registry=_registry(entry))


def test_returns_false_when_registry_omitted() -> None:
    entry = _entry(context_size=200000)
    request = _request(entry.namespaced_id)
    assert not matches(_atom("context_size", "gte", 1), request)


def test_returns_false_when_model_not_in_registry() -> None:
    entry = _entry(context_size=200000)
    request = _request("unknown-model")
    assert not matches(_atom("context_size", "gte", 1), request, registry=_registry(entry))


def test_resolves_bare_raw_id_when_unambiguous() -> None:
    entry = _entry(context_size=200000)
    request = _request("anthropic/claude-sonnet-4-6")  # bare raw_id, not namespaced
    assert matches(_atom("context_size", "gte", 200000), request, registry=_registry(entry))
