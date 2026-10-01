"""Independent stopped-cell review keeps historical records outside live types."""

import builtins
import importlib
import json
import os
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest
from operational_input_fixtures import candidates, manifests
from stopped_study_authorization_fixtures import verify as verify_root
from stopped_study_cell_fixtures import (
    cell_history,
    change_record,
    invented_cell_history,
    refresh_cell,
    verify_cell,
)
from study_run_record_fixtures import prepared
from test_stopped_study_authorization_review import _guard_observations

__all__ = [
    "candidates",
    "manifests",
    "prepared",
    "cell_history",
    "invented_cell_history",
]


_FORBIDDEN_TYPES = {
    "ExecutionBinding",
    "StudyExecutionBinding",
    "CellDescriptorPayloads",
    "RestoredOperationalCell",
    "VerifiedOperationalWorking",
    "VerifiedOperationalCell",
    "WorkerObservation",
    "OwnedProcessExit",
}


def _forbidden(*args, **kwargs):
    pytest.fail("historical cell verification cannot construct authority or perform IO")


def _guard_live_types(monkeypatch):
    for name in (
        "_operational_cell_results",
        "operational_cell_inputs",
        "operational_inputs",
        "study_execution",
        "execution_preflight",
        "study_operational_records",
        "stopped_study_cell",
    ):
        importlib.import_module(f"automated_phishing_detection.{name}")
    guarded = set()
    for module_name, module in tuple(sys.modules.items()):
        if not module_name.startswith("automated_phishing_detection."):
            continue
        for name, value in tuple(vars(module).items()):
            if not isinstance(value, type):
                continue
            if value.__name__ in _FORBIDDEN_TYPES or value.__name__.startswith(
                "Accepted"
            ):
                guarded.add(value.__name__)
                monkeypatch.setattr(module, name, _forbidden)
    assert _FORBIDDEN_TYPES <= guarded
    assert {"AcceptedOperationalInputs", "AcceptedOperationalRun"} <= guarded


def test_history_does_not_create_live_or_working_authority(cell_history, monkeypatch):
    _guard_observations(monkeypatch)
    _guard_live_types(monkeypatch)
    monkeypatch.setattr(builtins, "open", _forbidden)
    monkeypatch.setattr(Path, "open", _forbidden)
    monkeypatch.setattr(os, "open", _forbidden)
    monkeypatch.setattr(subprocess, "Popen", _forbidden)
    result = verify_cell(cell_history)
    assert result.attempt_payloads == cell_history.attempt_payloads
    assert result.input_payloads == cell_history.input_payloads


def _replace_service_pid(case, process_pid):
    for name in (
        "service-started.json",
        "service-process.json",
        "service-role.json",
        "service-ready.json",
        "service-cleanup.json",
    ):
        change_record(case, name, pid=process_pid)
    observation = json.loads(case.values["process-pair.json"])
    observation["service"]["pid"] = process_pid
    observation["cleanup_sha256"] = sha256(
        case.values["service-cleanup.json"]
    ).hexdigest()
    change_record(case, "process-pair.json", **observation)
    case.root.accounting["authorization_ledger"]["admissions"][-1]["launched_pid"] = (
        process_pid
    )
    refresh_cell(case)


def test_service_cannot_be_its_own_parent_after_all_record_joins_match(cell_history):
    parent = verify_root(cell_history.root).parent_pid
    _replace_service_pid(cell_history, parent)
    with pytest.raises(ValueError, match="^invalid_stopped_study_authorization$"):
        verify_root(cell_history.root)
    with pytest.raises(ValueError, match="^invalid_stopped_study_cell_history$"):
        verify_cell(cell_history)


def test_unlaunched_client_intent_is_retained_without_creating_a_launch(cell_history):
    previous_pins = cell_history.attempt_pins.copy()
    change_record(
        cell_history, "process-pair-intent.json", client_command_sha256="e" * 64
    )
    result = verify_cell(cell_history)
    payloads = dict(result.attempt_payloads)
    intent = json.loads(payloads["attempt/process-pair-intent.json"])
    observation = json.loads(payloads["attempt/process-pair.json"])
    assert intent["client_command_sha256"] == "e" * 64
    assert observation["client"]["pid"] is None
    assert observation["client"]["exit_observed"] is False
    assert (
        payloads["attempt/process-pair-intent.json"]
        == cell_history.values["process-pair-intent.json"]
    )
    with pytest.raises(ValueError, match="^invalid_stopped_study_cell_history$"):
        verify_cell(cell_history, expected_attempt_snapshot_sha256=previous_pins)


@pytest.mark.parametrize(
    "name",
    [
        "process-pair-intent.json",
        "service-intent.json",
        "service-started.json",
        "service-process.json",
        "service-role.json",
        "service-ready.json",
        "service-stop.json",
        "service-cleanup.json",
        "process-pair.json",
    ],
)
def test_process_records_cannot_use_the_receipt_codec(cell_history, name):
    assert cell_history.values[name].endswith(b"\n")
    cell_history.values[name] = cell_history.values[name][:-1]
    refresh_cell(cell_history)
    with pytest.raises(ValueError, match="^invalid_stopped_study_cell_history$"):
        verify_cell(cell_history)


@pytest.mark.parametrize("name", ["descriptor.json", "binding.json"])
def test_input_records_require_canonical_bytes(cell_history, name):
    assert cell_history.inputs[name].endswith(b"\n")
    cell_history.inputs[name] = cell_history.inputs[name][:-1]
    refresh_cell(cell_history)
    with pytest.raises(ValueError, match="^invalid_stopped_study_cell_history$"):
        verify_cell(cell_history)


def test_readiness_absence_retains_late_child_ready_file(cell_history):
    result = verify_cell(cell_history)
    payloads = dict(result.attempt_payloads)
    observation = json.loads(payloads["attempt/process-pair.json"])
    assert observation["readiness_sha256"] is None
    assert (
        payloads["attempt/service-ready.json"]
        == cell_history.values["service-ready.json"]
    )
    assert observation["failure"] == "parent_cancelled"
