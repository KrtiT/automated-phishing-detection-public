"""Closed saved accounting preserves failed history and never grants acceptance."""

import json
from dataclasses import replace

import pytest
from study_series_accounting_fixtures import (
    accounting_case,
    api,
    complete_ledger,
    segment,
    series,
    verify_segment,
)
from study_series_ledger_fixtures import (
    candidates,
    ledger,
    ledger_cell,
    manifests,
    observed,
    prefix_case,
    series_case,
    start,
)

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._study_history_snapshot_records import digest
from automated_phishing_detection.study_series_ledger import (
    SeriesCellStop,
    snapshot_series_ledger,
)

__all__ = [
    "accounting_case",
    "candidates",
    "manifests",
    "series_case",
    "prefix_case",
    "ledger_cell",
]


def test_complete_segment_requires_entire_fresh_suffix(accounting_case):
    case = accounting_case
    content = segment(
        case.prefix,
        ledger_bytes=complete_ledger(case.selected),
        status="complete",
        stage="segment_finalization",
    )
    value = verify_segment(case.prefix, content)
    assert value["status"] == "complete"
    assert value["ledger"]["cells"][0]["ordinal"] == 125


def test_accepted_prefix_can_fail_without_inventing_current_cell(ledger_cell):
    content = segment(ledger_cell.prefix, ledger_bytes=complete_ledger(ledger_cell))
    value = verify_segment(ledger_cell.prefix, content)
    assert value["ledger"]["cells"][0]["status"] == "accepted"
    assert value["ledger"]["cells"][1]["status"] == "unattempted"
    with pytest.raises(api().SeriesAccountingError):
        segment(
            ledger_cell.prefix,
            ledger_bytes=complete_ledger(ledger_cell),
            status="complete",
            stage="segment_finalization",
        )


def test_failed_initialization_retains_actual_segment_reservation(prefix_case):
    content = api().segment_accounting_bytes(
        prefix_case.binding,
        prefix_case.series,
        prefix_case.segment,
        status="failed",
        stage="input_retention",
    )
    value = api().verify_segment_accounting(
        content,
        prefix_case.binding,
        prefix_case.series,
        prefix_case.segment,
        expected_sha256=digest(content),
    )
    assert value["ledger"] is value["history_import_sha256"] is None
    assert value["segment_reservation_sha256"] == prefix_case.segment.reservation_sha256


def test_stopped_candidate_does_not_claim_holder_success(ledger_cell):
    current = ledger(ledger_cell.prefix)
    start(current, ledger_cell)
    observed(current, ledger_cell)
    current.stop_cell(
        SeriesCellStop(
            stage="finalization",
            candidate=ledger_cell.candidate,
            observation=ledger_cell.observation,
            pair_intent_bytes=ledger_cell.pair,
            publishing=True,
        )
    )
    value = verify_segment(
        ledger_cell.prefix,
        segment(ledger_cell.prefix, ledger_bytes=snapshot_series_ledger(current)),
    )
    slot = value["ledger"]["cells"][0]
    assert slot["status"] == "stopped" and slot["holders_closed"] is False
    assert len(slot["snapshot_sha256"]) == 36


def test_series_failure_preserves_completed_segment(accounting_case):
    case = accounting_case
    content = segment(
        case.prefix,
        ledger_bytes=complete_ledger(case.selected),
        status="complete",
        stage="segment_finalization",
    )
    value = json.loads(series(case, segment_bytes=content))
    assert value["status"] == "failed"
    assert value["segment"]["status"] == "complete"
    assert value["segment"]["accounting_sha256"] == digest(content)
    assert len(value["prior_attempts"]) == 4
    assert (
        value["prior_attempts"][-1]["disposition"] == "selected_entire_eligible_prefix"
    )


def test_series_unattempted_segment_has_only_null_pins(accounting_case):
    value = json.loads(series(accounting_case))
    assert value["segment"] == dict(
        ordinal=2, status="unattempted", reservation_sha256=None, accounting_sha256=None
    )
    with pytest.raises(api().SeriesAccountingError):
        series(accounting_case, status="complete")


def test_constructed_history_index_is_reauthenticated(accounting_case):
    case = accounting_case
    bad = replace(case.index, index_bytes=b"{}\n")
    with pytest.raises(api().SeriesAccountingError):
        api().series_accounting_bytes(
            case.prefix.binding, case.prefix.series, bad, status="failed"
        )


@pytest.mark.parametrize(
    "path,wrong",
    [
        (("schema_version",), True),
        (("unexpected",), None),
        (("stage",), "capacity_hold"),
        (("ledger", "end_ordinal"), 124),
        (("ledger", "parent_pid"), True),
        (("ledger", "cells", 0, "holders_closed"), False),
        (("ledger", "cells", 0, "snapshot_sha256"), {}),
        (("ledger", "admissions", 0, "accepted"), False),
        (("ledger", "admissions", 0, "exit_code"), True),
        (("ledger", "admissions", 0, "observation_recorded"), False),
        (("ledger", "admissions", 1, "launched_pid"), 321),
    ],
)
def test_rehashed_accounting_rejects_false_success(accounting_case, path, wrong):
    case = accounting_case
    value = json.loads(
        segment(
            case.prefix,
            ledger_bytes=complete_ledger(case.selected),
            status="complete",
            stage="segment_finalization",
        )
    )
    selected = value
    for key in path[:-1]:
        selected = selected[key]
    selected[path[-1]] = wrong
    with pytest.raises(api().SeriesAccountingError):
        verify_segment(case.prefix, canonical_bytes(value))
