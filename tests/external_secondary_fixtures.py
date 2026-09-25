"""Synthetic primary receipts and intercepted bound-secondary inference."""

import importlib
import importlib.util
from types import ModuleType, SimpleNamespace

import pytest
from external_producer_fixtures import prepared_external
from test_external_primary import external_session

from automated_phishing_detection.external_primary import (
    ExternalPrimaryScores,
    score_external_primary,
)

CHECKPOINTS = (
    *(
        f"secondary-tabular-{name}.json"
        for name in (
            "formatting",
            "permutation_42",
            "permutation_43",
            "permutation_44",
            "permutation_45",
            "permutation_46",
            "random_forest",
        )
    ),
    *(f"secondary-seed-{seed}.json" for seed in range(42, 47)),
)


@pytest.fixture
def module() -> ModuleType:
    name = "automated_phishing_detection.external_secondary"
    assert importlib.util.find_spec(name), "missing external secondary phase"
    return importlib.import_module(name)


def primary_scores(
    monkeypatch: pytest.MonkeyPatch, count: int
) -> ExternalPrimaryScores:
    session, unused_calls = external_session(monkeypatch)
    return score_external_primary(prepared_external(count), session)


@pytest.fixture
def phase(monkeypatch: pytest.MonkeyPatch, scoring: SimpleNamespace) -> SimpleNamespace:
    scoring.primary = primary_scores(monkeypatch, 2)
    scoring.events.clear()
    scoring.singletons.clear()
    return scoring
