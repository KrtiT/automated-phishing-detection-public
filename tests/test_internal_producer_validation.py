"""Bindings and completed rows are validated before they can become checkpoints."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from internal_retention_fixtures import COLUMNS, inputs, progress, run

from automated_phishing_detection import _internal_producer_checkpoints as checkpoints
from automated_phishing_detection import evaluation_producer
from automated_phishing_detection.selective_inference import InferenceCounts


@pytest.mark.parametrize(
    "fault", ["features", "decision", "counts", "audit", "secondary", "row_type"]
)
def test_malformed_primary_result_never_enters_complete_prefix(monkeypatch, fault):
    fixture, state = inputs(monkeypatch), progress()
    original = evaluation_producer._score_row

    def malformed(*args):
        row = original(*args)
        changes = {
            "features": {"features": list(row.features)},
            "decision": {"length_decision": float(row.length_decision)},
            "counts": {"inference_counts": InferenceCounts(True, 1, 1, 0)},
            "audit": {"length_scoring_audit_json": '{"private": 1}'},
            "secondary": {"secondary_tabular": []},
        }
        return (
            SimpleNamespace(**vars(row))
            if fault == "row_type"
            else replace(row, **changes[fault])
        )

    monkeypatch.setattr(evaluation_producer, "_score_row", malformed)
    with pytest.raises((ValueError, TypeError)):
        run(fixture, state)
    snapshot = json.loads(state.snapshot())
    assert snapshot["completed_primary_rows"] == []
    assert snapshot["started_primary_position"] == 1
    assert snapshot["inference_counts"]["completed_requests"] == 1


@pytest.mark.parametrize(
    "fault", ["primary_bytes", "tabular_bytes", "seed_bytes", "threshold"]
)
def test_invalid_model_byte_identity_rejects_before_any_forward(monkeypatch, fault):
    fixture, state = inputs(monkeypatch), progress()
    bound = fixture.session.secondary
    if fault == "primary_bytes":
        fixture.session.primary.models.gmm_artifact_bytes = b"changed"
    elif fault == "tabular_bytes":
        member = replace(bound.tabular[0], artifact_sha256="f" * 64)
        bound = replace(bound, tabular=(member, *bound.tabular[1:]))
    elif fault == "seed_bytes":
        member = replace(bound.seeds[1], _weights_bytes=b"changed")
        bound = replace(bound, seeds=(bound.seeds[0], member, *bound.seeds[2:]))
    else:
        bound = replace(bound, stage1_threshold=0.4)
    fixture.session = replace(fixture.session, secondary=bound)
    with pytest.raises(ValueError):
        run(fixture, state)
    assert fixture.session.primary.scorer.urls == []
    assert fixture.secondary_calls == []


def test_live_binding_mutation_is_rejected_without_rewriting_early_snapshot(
    monkeypatch,
):
    fixture, state = inputs(monkeypatch), progress()
    frozen = []

    def retain(name, content):
        if name == "bindings.json":
            frozen.append(content)
        if name == "routing.json":
            fixture.session.primary.models.monitor_boundary += 1

    with pytest.raises(
        evaluation_producer.EvaluationProducerError, match="binding changed"
    ):
        run(fixture, state, retain)
    assert state.outputs["bindings.json"] is frozen[0]
    assert "predictions.jsonl" in state.outputs


def test_counts_observation_never_replaces_original_failure():
    state = progress()

    class MissingCounts:
        @property
        def counts(self):
            raise KeyboardInterrupt("private-observation-message")

    state.observe_counts(MissingCounts(), suppress_interruptions=True)
    snapshot = json.loads(state.snapshot())
    assert snapshot["inference_counts"] is None
    assert snapshot["inference_counts_reason"] == "physical_counts_unavailable"


def test_secondary_return_must_equal_the_retained_completed_columns(monkeypatch):
    fixture, state = inputs(monkeypatch), progress()
    original = evaluation_producer.score_bound_secondary

    def changed(*args, **kwargs):
        scoring = original(*args, **kwargs)
        first = scoring.rows[0]
        tabular = replace(first.tabular[0], probability=0.123456789)
        first = replace(first, tabular=(tabular, *first.tabular[1:]))
        return replace(scoring, rows=(first, *scoring.rows[1:]))

    monkeypatch.setattr(evaluation_producer, "score_bound_secondary", changed)
    with pytest.raises(ValueError, match="columns_differ"):
        run(fixture, state)
    assert len(state.columns) == 12
    assert "predictions.jsonl" not in state.outputs


@pytest.mark.parametrize("bad_primary", [None, object()])
def test_invalid_session_cannot_replace_its_first_input_error(monkeypatch, bad_primary):
    fixture, state = inputs(monkeypatch), progress()
    fixture.session = replace(fixture.session, primary=bad_primary)
    with pytest.raises(evaluation_producer.EvaluationProducerError):
        run(fixture, state)
    assert json.loads(state.snapshot())["completed_primary_rows"] == []


def test_invalid_completed_column_is_not_reported_as_unattempted(monkeypatch):
    fixture, state = inputs(monkeypatch), progress()
    original = evaluation_producer.score_bound_secondary

    def malformed(*args, on_completed_column):
        def notify(column):
            return on_completed_column(replace(column, singleton_calls=-1))

        return original(*args, on_completed_column=notify)

    monkeypatch.setattr(evaluation_producer, "score_bound_secondary", malformed)
    with pytest.raises(ValueError):
        run(fixture, state)
    snapshot = json.loads(state.snapshot())
    assert len(fixture.secondary_calls) == 1
    assert snapshot["unattempted_secondary_members"] == list(COLUMNS[1:])
    assert (
        snapshot["incomplete_member_counts_reason"]
        == "incomplete_member_counts_unavailable"
    )


def test_validated_column_survives_checkpoint_encoding_failure(monkeypatch):
    fixture, state = inputs(monkeypatch), progress()

    def failed(*args, **kwargs):
        raise ValueError("private-encoding-failure")

    monkeypatch.setattr(checkpoints, "column_bytes", failed)
    with pytest.raises(ValueError):
        run(fixture, state)
    snapshot = json.loads(state.snapshot())
    assert len(snapshot["completed_secondary_columns"]) == 1
    assert snapshot["completed_secondary_columns"][0]["singleton_calls"] == 4
    assert snapshot["unattempted_secondary_members"] == list(COLUMNS[1:])
    assert COLUMNS[0] not in state.outputs
