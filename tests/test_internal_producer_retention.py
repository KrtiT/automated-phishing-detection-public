"""Internal producer retention is ordered and preserves its scientific bytes."""

import json
from hashlib import sha256

import pytest
from internal_retention_fixtures import COLUMNS, ORDER, inputs, progress, run

from automated_phishing_detection import bound_secondary, evaluation_producer


def test_success_retains_eighteen_boundaries_without_changing_scientific_bytes(
    monkeypatch,
):
    fixture, state = inputs(monkeypatch), progress()
    writes = []
    result = run(fixture, state, lambda *item: writes.append(item))
    assert tuple(name for name, content in writes) == ORDER
    assert dict(writes) == state.outputs
    assert all(
        state.outputs[name] is content
        for name, content in result.private_outputs.items()
    )
    assert {
        name: sha256(content).hexdigest()
        for name, content in result.private_outputs.items()
    } == {
        "predictions.jsonl": "a1b1916abaa93bd81967d9d731f738635be442f194f43bede2c9035b022d5aab",
        "routing.json": "c486820ee9fb27c203fd8a39e5d4b0a23f982c66bba64c7ac88ed0be541b195c",
        "manifests.json": "a3495f4b4972c5b6c96e7d56ebfbc87c650d412a0774452fc4d79fc22c654cf8",
        "bindings.json": "dfd677e6eddb69b09a42758a495b54730618be0c76446647714c87b424ff79b1",
    }
    assert sha256(
        evaluation_producer._json_bytes(result.public_summary)
    ).hexdigest() == (
        "191405e975fbb57889a935fd5f53c7eec1acbf91afd8e0913f7c382634f1185d"
    )
    assert len(fixture.session.primary.scorer.urls) == 4
    assert len(fixture.secondary_calls) == 11


def test_primary_checkpoint_and_each_column_precede_the_next_member(monkeypatch):
    fixture, state = inputs(monkeypatch), progress()
    original = bound_secondary._completed_columns
    writes = {}

    def observed(*arguments):
        assert "primary-completion.json" in writes
        for index, column in enumerate(original(*arguments)):
            assert all(name in writes for name in COLUMNS[:index])
            yield column

    monkeypatch.setattr(bound_secondary, "_completed_columns", observed)
    run(fixture, state, lambda name, content: writes.__setitem__(name, content))
    receipt = json.loads(writes["primary-completion.json"])
    assert receipt == {
        "schema_version": 1,
        "phase": "internal_primary",
        "row_count": 4,
        "partition_sha256": fixture.prepared.partition_sha256,
        "bindings_sha256": sha256(writes["bindings.json"]).hexdigest(),
        "primary_scores_sha256": sha256(writes["primary-scores.jsonl"]).hexdigest(),
        "inference_counts": json.loads(state.snapshot())["inference_counts"],
    }


def test_joined_predictions_are_retained_before_population_evaluation(monkeypatch):
    fixture, state = inputs(monkeypatch), progress()

    def failed(**kwargs):
        assert "predictions.jsonl" in state.outputs
        assert all(name in state.outputs for name in COLUMNS)
        raise ValueError("private-derived-message")

    monkeypatch.setattr(
        evaluation_producer.hypothesis_evaluation, "evaluate_primary", failed
    )
    with pytest.raises(ValueError, match="private-derived-message"):
        run(fixture, state)
    assert "routing.json" not in state.outputs


@pytest.mark.parametrize("retain", [False, 1, "private-callback"])
def test_bad_callback_is_rejected_before_any_forward(monkeypatch, retain):
    fixture, state = inputs(monkeypatch), progress()
    with pytest.raises(evaluation_producer.EvaluationProducerError):
        run(fixture, state, retain)
    assert fixture.session.primary.scorer.urls == []


def test_progress_cannot_be_reused_for_another_attempt(monkeypatch):
    fixture, state = inputs(monkeypatch), progress()
    run(fixture, state)
    next_fixture = inputs(monkeypatch)
    with pytest.raises(evaluation_producer.EvaluationProducerError):
        run(next_fixture, state)
    assert next_fixture.session.primary.scorer.urls == []
