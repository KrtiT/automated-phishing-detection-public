"""Construction, launched PID, and owned exit remain separate observations."""

import json

import adopted_study_fixtures as fixtures
import pytest
from study_operational_fixtures import compact
from study_run_record_fixtures import capacity, prepared

from automated_phishing_detection import _study_run_records as records
from automated_phishing_detection import study_operational_records
from automated_phishing_detection._checkpoint_codec import canonical_bytes

__all__ = ["prepared"]


def ledger(prepared):
    module = fixtures.api("_adopted_study_ledger")
    case = type(
        "Case",
        (),
        dict(
            binding=prepared.binding, profile=prepared.profile, paths=None, deadlines={}
        ),
    )()
    value = module.AdmissionLedger(
        fixtures.authorization(case), prepared.attempt, b"intent"
    )
    return value


def test_no_admission_without_confirmed_capacity_barrier(prepared):
    value = ledger(prepared)
    with pytest.raises(ValueError):
        value.issue("internal", ("invented",))
    assert json.loads(value.snapshot())["admissions"] == []


def test_shortage_barrier_cannot_become_admission(prepared):
    value = ledger(prepared)
    barrier, held = records.prediction_barrier(
        prepared.preparation, execution=prepared.execution
    )
    assert held
    with pytest.raises(ValueError):
        value.barrier_retained(prepared.preparation, barrier)


def test_issued_frame_is_retained_without_invented_launch(prepared):
    value = ledger(prepared)
    preparation = capacity(prepared)
    barrier, held = records.prediction_barrier(
        preparation, execution=prepared.execution
    )
    assert not held
    value.barrier_retained(preparation, barrier)
    admission = value.issue("internal", ("invented",))
    entry = json.loads(value.snapshot())["admissions"][0]
    assert entry["issued"] is True
    assert entry["launched_pid"] is None
    assert entry["exit_observed"] is False
    assert entry["exit_code"] is None
    assert entry["accepted"] is False
    assert (
        fixtures.api("_adopted_study_records").decoded(entry["frame_bytes"])
        == admission.frame.canonical_bytes
    )
    assert entry["frame_sha256"] == admission.frame.sha256
    with pytest.raises(ValueError):
        value.issue("external", ("invented",), predecessor_sha256="a" * 64)
    with pytest.raises(ValueError):
        value.issue("internal", ("invented",))


def test_cell_acceptance_requires_original_observation_and_binding(prepared):
    value = ledger(prepared)
    retained = compact(study_operational_records, 1)
    value.current_cell = retained.cell
    value.entries = [
        {
            "role": role,
            "cell_ordinal": 1,
            "exit_observed": True,
            "exit_code": 0,
            "launched_pid": 999,
            "command_sha256": "a" * 64,
        }
        for role in ("service", "client")
    ]
    intent = canonical_bytes(
        {
            "schema_version": 1,
            "reservation_sha256": "1" * 64,
            "service_command_sha256": "a" * 64,
            "client_command_sha256": "a" * 64,
            "deadlines": {},
        }
    )
    with pytest.raises(ValueError):
        value.cell_accepted(retained.cell, retained, pair_intent_bytes=intent)
    assert value.completed_cells == 0


def test_source_acceptance_requires_matching_actual_owned_observation(prepared):
    from types import SimpleNamespace

    from automated_phishing_detection._owned_process_exit import OwnedProcessExit
    from automated_phishing_detection.owned_worker import WorkerObservation

    value = ledger(prepared)
    value.entries = [
        {
            "role": "internal",
            "accepted": False,
            "command_sha256": "a" * 64,
            "launched_pid": 123,
            "exit_observed": False,
            "exit_code": None,
        }
    ]
    completion = SimpleNamespace(
        worker=WorkerObservation(
            "a" * 64, OwnedProcessExit(123, True, 0), "b" * 64, "c" * 64
        )
    )
    with pytest.raises(ValueError):
        value.internal_accepted(completion)
    assert value.handoff_sha256 is None


def test_other_root_barrier_cannot_issue_internal_admission(prepared):
    value = ledger(prepared)
    preparation = capacity(prepared)
    barrier, unused = records.prediction_barrier(
        preparation, execution=prepared.execution
    )
    changed = json.loads(barrier)
    changed["execution"]["reservation_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        value.barrier_retained(preparation, canonical_bytes(changed))
