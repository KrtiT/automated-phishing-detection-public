"""Invented real children join exact ledger bytes with actual owned PID and exit."""

import json

import pytest
from study_admission_fixtures import child_command
from study_run_record_fixtures import capacity, prepared
from test_adopted_study_ledger import ledger

from automated_phishing_detection import _study_run_records as records
from automated_phishing_detection.owned_worker import (
    WorkerExecutionError,
    _observe_study_worker,
)

__all__ = ["prepared"]


def admitted(prepared, command):
    value = ledger(prepared)
    preparation = capacity(prepared)
    barrier, unused = records.prediction_barrier(
        preparation, execution=prepared.execution
    )
    value.barrier_retained(preparation, barrier)
    return value, value.issue("internal", command)


def test_real_owned_child_records_only_actual_pid_and_exit(prepared):
    command = child_command()
    value, admission = admitted(prepared, command)
    observation = _observe_study_worker(command, admission)
    entry = json.loads(value.snapshot())["admissions"][0]
    assert entry["launched_pid"] == observation.exit.pid
    assert entry["exit_observed"] is True and entry["exit_code"] == 0
    assert entry["accepted"] is False
    assert entry["frame_sha256"] == admission.frame.sha256


def test_failed_owned_launch_retains_issuance_without_synthetic_pid(prepared):
    command = ("/invented/not-an-executable",)
    value, admission = admitted(prepared, command)
    with pytest.raises(WorkerExecutionError):
        _observe_study_worker(command, admission)
    entry = json.loads(value.snapshot())["admissions"][0]
    assert entry["issued"] is True
    assert entry["launched_pid"] is None
    assert entry["exit_observed"] is False and entry["exit_code"] is None
    assert entry["accepted"] is False
