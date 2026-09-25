"""Invented existing-producer observations for shared saved-score validation."""

import json

import pytest
from test_evaluation_producer import parse, synthetic_session
from test_saved_evidence import expect_synthetic_binding

from automated_phishing_detection import evaluation_producer, saved_evidence


@pytest.fixture
def saved_scores(monkeypatch):
    prepared = parse(evaluation_producer)
    session, *_ = synthetic_session(evaluation_producer, monkeypatch)
    produced = evaluation_producer.produce_internal_evidence(prepared, session)
    outputs = produced.private_outputs
    expect_synthetic_binding(saved_evidence, monkeypatch, outputs["bindings.json"])
    return (
        [json.loads(line) for line in outputs["predictions.jsonl"].splitlines()],
        json.loads(outputs["bindings.json"]),
    )


def validate(row, bindings):
    checker = getattr(saved_evidence, "_validate_score_row", None)
    assert callable(checker), "missing shared saved-score validator"
    return checker(row, bindings)
