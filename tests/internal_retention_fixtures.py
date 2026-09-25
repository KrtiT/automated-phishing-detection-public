"""Invented primary scores and real completed-column orchestration only."""

from types import SimpleNamespace

from external_composition_fixtures import _secondary_scorers
from test_evaluation_producer import parse, synthetic_session

from automated_phishing_detection import bound_secondary, evaluation_producer

COLUMNS = (
    *(f"secondary-tabular-{name}.json" for name in bound_secondary._TABULAR_NAMES),
    *(f"secondary-seed-{seed}.json" for seed in bound_secondary._SEEDS),
)
ORDER = (
    "bindings.json",
    "manifests.json",
    "primary-scores.jsonl",
    "primary-completion.json",
    *COLUMNS,
    "predictions.jsonl",
    "routing.json",
)


def inputs(monkeypatch):
    prepared = parse(evaluation_producer)
    session, *primary_calls = synthetic_session(evaluation_producer, monkeypatch)
    secondary_calls = []
    _secondary_scorers(monkeypatch, secondary_calls)
    monkeypatch.setattr(
        evaluation_producer,
        "score_bound_secondary",
        bound_secondary.score_bound_secondary,
    )
    return SimpleNamespace(
        prepared=prepared,
        session=session,
        primary_calls=primary_calls,
        secondary_calls=secondary_calls,
    )


def progress():
    carrier = getattr(evaluation_producer, "InternalProgress", None)
    assert callable(carrier), "missing parent-owned internal progress"
    return carrier()


def run(fixture, state, retain=None):
    return evaluation_producer.produce_internal_evidence(
        fixture.prepared, fixture.session, progress=state, retain=retain
    )
