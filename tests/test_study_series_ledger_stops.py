"""One explicit stopped slot retains partial facts without false acceptance."""

import base64
from dataclasses import FrozenInstanceError

import pytest
from study_series_ledger_fixtures import (
    api,
    candidates,
    commands,
    ledger,
    ledger_cell,
    manifests,
    observed,
    prefix_case,
    series_case,
    snapshot,
    start,
)
from test_study_series_ledger_ordering import rejection

__all__ = ["candidates", "manifests", "series_case", "prefix_case", "ledger_cell"]


@pytest.mark.parametrize("stage", ["validation", "selection", "reservation"])
def test_stop_before_reservation_is_not_a_fabricated_capacity_hold(prefix_case, stage):
    current = ledger(prefix_case)
    current.stop_cell(api().SeriesCellStop(stage))
    value = snapshot(current)
    slot = value["cells"][0]
    assert slot["status"] == "stopped" and slot["stage"] == stage
    assert slot["reservation_sha256"] is None
    assert slot["descriptor_bytes"] is None and slot["observation_bytes"] is None
    assert slot["holders_closed"] is False
    assert value["admissions"] == []


def test_active_snapshot_rejection_does_not_falsely_stop_or_poison_acceptance(
    ledger_cell,
):
    current = ledger(ledger_cell.prefix)
    start(current, ledger_cell)
    with rejection():
        snapshot(current)
    observed(current, ledger_cell)
    current.accept_cell(
        ledger_cell.candidate, ledger_cell.observation, ledger_cell.pair
    )
    assert snapshot(current)["cells"][0]["status"] == "accepted"


def test_late_holder_failure_keeps_complete_candidate_but_no_acceptance(ledger_cell):
    current = ledger(ledger_cell.prefix)
    start(current, ledger_cell)
    observed(current, ledger_cell)
    stopped = api().SeriesCellStop(
        "finalization",
        observation=ledger_cell.observation,
        pair_intent_bytes=ledger_cell.pair,
        candidate=ledger_cell.candidate,
        publishing=True,
    )
    current.stop_cell(stopped)
    value = snapshot(current)
    slot = value["cells"][0]
    assert slot["status"] == "stopped" and slot["holders_closed"] is False
    assert slot["publishing"] is True and len(slot["snapshot_sha256"]) == 36
    assert base64.b64decode(slot["observation_bytes"]) == ledger_cell.observation.record
    assert all(not entry["accepted"] for entry in value["admissions"])
    with pytest.raises(FrozenInstanceError):
        stopped.publishing = False
    with rejection():
        current.accept_cell(
            ledger_cell.candidate, ledger_cell.observation, ledger_cell.pair
        )


def test_partial_process_progress_survives_missing_pair_publication(ledger_cell):
    current = ledger(ledger_cell.prefix)
    start(current, ledger_cell)
    current.stop_cell(
        api().SeriesCellStop(
            "observation", progress_bytes=ledger_cell.observation.record
        )
    )
    slot = snapshot(current)["cells"][0]
    assert base64.b64decode(slot["progress_bytes"]) == ledger_cell.observation.record
    assert slot["observation_bytes"] is None and slot["snapshot_sha256"] is None


@pytest.mark.parametrize("stage", ["private_error_text", "", None, True, "complete"])
def test_only_closed_symbolic_failure_stages(prefix_case, stage):
    current = ledger(prefix_case)
    with rejection():
        current.stop_cell(api().SeriesCellStop(stage))


@pytest.mark.parametrize(
    "field",
    [
        "attempt",
        "descriptor_bytes",
        "binding_bytes",
        "observation",
        "pair_intent_bytes",
        "progress_bytes",
        "candidate",
        "publishing",
    ],
)
def test_wrong_partial_fact_is_not_silently_dropped(ledger_cell, field):
    current = ledger(ledger_cell.prefix)
    start(current, ledger_cell)
    with rejection():
        current.stop_cell(api().SeriesCellStop("observation", **{field: "wrong fact"}))


def test_stop_is_single_use_and_retained_snapshot_is_immutable(ledger_cell):
    current = ledger(ledger_cell.prefix)
    start(current, ledger_cell)
    current.stop_cell(api().SeriesCellStop("observation"))
    content = api().snapshot_series_ledger(current)
    view = snapshot(current)
    view["cells"][0]["status"] = "accepted"
    assert api().snapshot_series_ledger(current) == content
    with rejection():
        current.stop_cell(api().SeriesCellStop("completion"))
    assert api().snapshot_series_ledger(current) == content


def test_returned_reservation_before_start_retains_only_obtained_inputs(ledger_cell):
    current = ledger(ledger_cell.prefix)
    current.stop_cell(
        api().SeriesCellStop(
            "input_retention",
            attempt=ledger_cell.attempt,
            descriptor_bytes=ledger_cell.selected.descriptor_bytes,
        )
    )
    slot = snapshot(current)["cells"][0]
    assert slot["reservation_sha256"] == ledger_cell.attempt.reservation_sha256
    assert (
        base64.b64decode(slot["descriptor_bytes"])
        == ledger_cell.selected.descriptor_bytes
    )
    assert slot["binding_bytes"] is None


def test_actual_cleanup_observation_after_stop_preserves_earlier_snapshot(ledger_cell):
    current = ledger(ledger_cell.prefix)
    start(current, ledger_cell)
    admission = current.issue("service", commands(ledger_cell)[0])
    admission.on_launched(321)
    current.stop_cell(api().SeriesCellStop("observation"))
    before = api().snapshot_series_ledger(current)
    admission.on_observed(True, 0)
    after = snapshot(current)
    assert after["admissions"][0]["observation_recorded"] is True
    assert api().snapshot_series_ledger(current) != before
