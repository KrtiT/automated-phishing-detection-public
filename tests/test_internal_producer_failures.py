"""Completed scientific observations survive catchable scoring and writer failure."""

import asyncio
import base64
import json

import pytest
from internal_retention_fixtures import COLUMNS, ORDER, inputs, progress, run

from automated_phishing_detection import bound_secondary, evaluation_producer


@pytest.mark.parametrize(
    "exception_type",
    [RuntimeError, KeyboardInterrupt, asyncio.CancelledError, SystemExit],
)
def test_primary_prefix_and_actual_counters_survive_interruption(
    monkeypatch, exception_type
):
    fixture, state = inputs(monkeypatch), progress()
    original = evaluation_producer._score_row
    failure = exception_type("private-row-message")

    def score(record, session, thresholds, position):
        if position == 3:
            raise failure
        return original(record, session, thresholds, position)

    monkeypatch.setattr(evaluation_producer, "_score_row", score)
    with pytest.raises(exception_type) as caught:
        run(fixture, state)
    snapshot = json.loads(state.snapshot())
    assert caught.value is failure
    assert snapshot["started_primary_position"] == 3
    assert len(snapshot["completed_primary_rows"]) == 2
    assert snapshot["inference_counts"]["transformer_forward_attempts"] == 2
    assert "primary-scores.jsonl" not in state.outputs
    assert fixture.secondary_calls == []


def test_monitor_failure_does_not_invent_counts_from_completed_row_prefix(monkeypatch):
    fixture, state = inputs(monkeypatch), progress()
    original = evaluation_producer._monitor_scores

    def monitor(*args):
        if len(fixture.session.primary.scorer.urls) == 3:
            raise ValueError("private-monitor-failure")
        return original(*args)

    monkeypatch.setattr(evaluation_producer, "_monitor_scores", monitor)
    with pytest.raises(ValueError):
        run(fixture, state)
    snapshot = json.loads(state.snapshot())
    assert len(snapshot["completed_primary_rows"]) == 2
    assert snapshot["inference_counts"]["completed_requests"] == 3


@pytest.mark.parametrize("failed_index", range(12))
def test_failure_in_each_secondary_member_preserves_all_complete_predecessors(
    monkeypatch, failed_index
):
    fixture, state = inputs(monkeypatch), progress()
    original = bound_secondary._completed_columns

    def columns(*args):
        for index, column in enumerate(original(*args)):
            if index == failed_index:
                raise RuntimeError("private-member-failure")
            yield column

    monkeypatch.setattr(bound_secondary, "_completed_columns", columns)
    with pytest.raises(RuntimeError):
        run(fixture, state)
    snapshot = json.loads(state.snapshot())
    assert "primary-completion.json" in state.outputs
    assert (
        tuple(name for name in COLUMNS if name in state.outputs)
        == COLUMNS[:failed_index]
    )
    assert snapshot["next_expected_member"] == COLUMNS[failed_index]
    assert snapshot["incomplete_member_counts"] is None
    assert (
        snapshot["incomplete_member_counts_reason"]
        == "incomplete_member_counts_unavailable"
    )


@pytest.mark.parametrize("failed_name", ORDER)
def test_ambiguous_writer_failure_caches_bytes_and_never_retries(
    monkeypatch, failed_name
):
    fixture, state = inputs(monkeypatch), progress()
    writes = []

    def retain(name, content):
        writes.append((name, content))
        if name == failed_name:
            raise OSError("private-writer-message")

    with pytest.raises(OSError):
        run(fixture, state, retain)
    snapshot = json.loads(state.snapshot())
    assert (
        tuple(name for name, content in writes) == ORDER[: ORDER.index(failed_name) + 1]
    )
    assert snapshot["failed_checkpoint"] == failed_name
    assert snapshot["retention_status"] == "failed_or_ambiguous"
    assert {
        name: base64.b64decode(content)
        for name, content in snapshot["retained_checkpoints_base64"].items()
    } == dict(writes)
