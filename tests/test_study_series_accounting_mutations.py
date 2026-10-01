"""Rehashed saved facts cannot acquire different context or successful status."""

import base64
import json

import pytest
from study_series_accounting_fixtures import (
    accounting_case,
    api,
    complete_ledger,
    completed_accounting,
    prefix_arguments,
    segment,
    verify_segment,
)
from study_series_ledger_fixtures import (
    candidates,
    commands,
    ledger,
    ledger_cell,
    manifests,
    prefix_case,
    series_case,
    start,
)

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._operational_process_records import _bytes
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
    "completed_accounting",
]


def encoded(value):
    return base64.b64encode(value).decode("ascii")


@pytest.mark.parametrize(
    "field",
    [
        "parent_pid",
        "profile_sha256",
        "envelope_sha256",
        "command_sha256",
        "series_reservation_sha256",
        "segment_reservation_sha256",
        "intent_sha256",
        "predecessor_sha256",
        "accepted_inputs_sha256",
        "cell_binding_sha256",
        "origin_reservation_sha256",
        "history_index_sha256",
        "segment_ordinal",
        "cell_ordinal",
        "role",
    ],
)
def test_rehashed_frame_cannot_change_context(
    accounting_case, completed_accounting, field
):
    value = json.loads(completed_accounting)
    entry = value["ledger"]["admissions"][0]
    frame = json.loads(base64.b64decode(entry["frame_bytes"]))
    frame[field] = frame[field] + 1 if type(frame[field]) is int else "0" * 64
    content = canonical_bytes(frame)[:-1]
    entry.update(frame_bytes=encoded(content), frame_sha256=digest(content))
    with pytest.raises(api().SeriesAccountingError):
        verify_segment(accounting_case.prefix, canonical_bytes(value))


@pytest.mark.parametrize(
    "name",
    [
        "attempt/service-ready.json",
        "attempt/service-cleanup.json",
        "attempt/service-process.json",
        "attempt/client-started.json",
    ],
)
def test_complete_snapshot_pins_join_process_facts(
    accounting_case, completed_accounting, name
):
    value = json.loads(completed_accounting)
    pins = value["ledger"]["cells"][0]["snapshot_sha256"]
    pins[name] = "0" * 64
    pins[name.replace("attempt/", "attempt/evidence/")] = "0" * 64
    with pytest.raises(api().SeriesAccountingError):
        verify_segment(accounting_case.prefix, canonical_bytes(value))


def test_forced_stopped_progress_preserves_actual_signal_and_unknown_exit(ledger_cell):
    selected = ledger_cell
    current = ledger(selected.prefix)
    start(current, selected)
    with current.issue("service", commands(selected)[0]) as admission:
        admission.launched(321)
        admission.observed(False, None)
    progress = json.loads(selected.observation.record)
    progress.update(status="failed", failure="service_exit_unsuccessful")
    progress["service"].update(
        exit_code=None, exit_observed=False, forced=True, signals=[15, 9]
    )
    progress["client"].update(
        pid=None,
        exit_code=None,
        exit_observed=False,
        stdout_sha256=None,
        stderr_sha256=None,
    )
    current.stop_cell(
        SeriesCellStop(stage="observation", progress_bytes=_bytes(progress))
    )
    value = verify_segment(
        selected.prefix,
        segment(selected.prefix, ledger_bytes=snapshot_series_ledger(current)),
    )
    assert value["ledger"]["admissions"][0]["observation_recorded"] is True
    assert value["ledger"]["cells"][0]["status"] == "stopped"


def test_initialized_import_is_retained_when_ledger_initialization_fails(prefix_case):
    content = api().segment_accounting_bytes(
        prefix_case.binding,
        prefix_case.series,
        prefix_case.segment,
        status="failed",
        stage="input_retention",
        **prefix_arguments(prefix_case),
    )
    value = verify_segment(prefix_case, content)
    assert value["history_import_sha256"] == digest(prefix_case.imported)
    assert value["ledger"] is None


def test_preexecution_stage_cannot_contain_accepted_cells(
    accounting_case, completed_accounting
):
    value = json.loads(completed_accounting)
    value.update(status="failed", stage="validation")
    with pytest.raises(api().SeriesAccountingError):
        verify_segment(accounting_case.prefix, canonical_bytes(value))


@pytest.mark.parametrize("mutation", ["reorder", "duplicate", "truncate", "extend"])
def test_suffix_roster_is_exact(ledger_cell, mutation):
    value = json.loads(
        segment(ledger_cell.prefix, ledger_bytes=complete_ledger(ledger_cell))
    )
    cells = value["ledger"]["cells"]
    if mutation == "reorder":
        cells[1], cells[2] = cells[2], cells[1]
    elif mutation == "duplicate":
        cells[2] = cells[1]
    elif mutation == "truncate":
        cells.pop()
    else:
        cells.append(cells[-1])
    with pytest.raises(api().SeriesAccountingError):
        verify_segment(ledger_cell.prefix, canonical_bytes(value))


@pytest.mark.parametrize(
    "field",
    ["descriptor_bytes", "binding_bytes", "observation_bytes", "pair_intent_bytes"],
)
def test_base64_must_be_canonical(accounting_case, completed_accounting, field):
    value = json.loads(completed_accounting)
    value["ledger"]["cells"][0][field] += "\n"
    with pytest.raises(api().SeriesAccountingError):
        verify_segment(accounting_case.prefix, canonical_bytes(value))
