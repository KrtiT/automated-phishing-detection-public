"""Counter observations never swallow a first interruption or replace an earlier one."""

import asyncio
import json

import pytest
from internal_retention_fixtures import inputs, progress, run

from automated_phishing_detection import evaluation_producer


def interrupted_counts(monkeypatch, scorer, failure):
    original = type(scorer).counts.fget
    armed = []

    def counts(current):
        if armed:
            raise failure
        return original(current)

    monkeypatch.setattr(type(scorer), "counts", property(counts))
    return armed


@pytest.mark.parametrize(
    "exception_type", [KeyboardInterrupt, asyncio.CancelledError, SystemExit]
)
def test_first_count_interruption_preserves_validated_row_and_stops(
    monkeypatch, exception_type
):
    fixture, state = inputs(monkeypatch), progress()
    failure = exception_type("private-count-message")
    armed = interrupted_counts(monkeypatch, fixture.session.primary.scorer, failure)
    original = evaluation_producer._validate_completed_primary

    def validated(*args):
        original(*args)
        armed.append(True)

    monkeypatch.setattr(evaluation_producer, "_validate_completed_primary", validated)
    with pytest.raises(exception_type) as caught:
        run(fixture, state)
    snapshot = json.loads(state.snapshot())
    assert caught.value is failure
    assert len(snapshot["completed_primary_rows"]) == 1
    assert snapshot["started_primary_position"] is None
    assert snapshot["inference_counts"] is None
    assert len(fixture.session.primary.scorer.urls) == 1
    assert fixture.secondary_calls == []


@pytest.mark.parametrize("original_type", [ValueError, KeyboardInterrupt])
def test_failure_observation_preserves_only_an_earlier_interruption(
    monkeypatch, original_type
):
    fixture, state = inputs(monkeypatch), progress()
    original_failure = original_type("private-score-message")
    counter_failure = asyncio.CancelledError("private-count-message")
    armed = interrupted_counts(
        monkeypatch, fixture.session.primary.scorer, counter_failure
    )

    def failed(*args):
        armed.append(True)
        raise original_failure

    monkeypatch.setattr(evaluation_producer, "_score_row", failed)
    expected = (
        counter_failure if isinstance(original_failure, Exception) else original_failure
    )
    with pytest.raises(type(expected)) as caught:
        run(fixture, state)
    assert caught.value is expected
    snapshot = json.loads(state.snapshot())
    assert snapshot["completed_primary_rows"] == []
    assert snapshot["inference_counts"] is None
    assert snapshot["inference_counts_reason"] == "physical_counts_unavailable"
    assert fixture.secondary_calls == []
