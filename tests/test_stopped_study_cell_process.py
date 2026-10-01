"""Recorded first failure, process state and lifecycle remain original and closed."""

import json

import pytest
from operational_input_fixtures import candidates, manifests
from stopped_study_cell_fixtures import (
    cell_history,
    change_record,
    invented_cell_history,
    verify_cell,
)
from study_run_record_fixtures import prepared

__all__ = [
    "candidates",
    "manifests",
    "prepared",
    "cell_history",
    "invented_cell_history",
]


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("unknown", True),
        ("status", "observed"),
        ("failure", "startup_timeout"),
        ("failure", "service_cleanup_failed"),
        ("failure", "process_pair_failed"),
        ("research_accepted", True),
        ("record_failures", ["service-ready.json"]),
        ("stop_sent", False),
        ("readiness_sha256", "0" * 64),
        ("cleanup_sha256", "0" * 64),
        ("reservation_sha256", "0" * 64),
    ],
)
def test_pair_contract_rejects_other_outcomes(cell_history, field, value):
    change_record(cell_history, "process-pair.json", **{field: value})
    with pytest.raises(ValueError):
        verify_cell(cell_history)


@pytest.mark.parametrize(
    "field,value",
    [
        ("pid", 999),
        ("exit_code", 1),
        ("exit_observed", False),
        ("forced", True),
        ("signals", [15]),
        ("stdout_sha256", "bad"),
        ("stderr_sha256", None),
        ("unknown", True),
    ],
)
def test_service_must_record_clean_observed_exit(cell_history, field, value):
    pair = json.loads(cell_history.values["process-pair.json"])
    pair["service"][field] = value
    change_record(cell_history, "process-pair.json", **pair)
    with pytest.raises(ValueError):
        verify_cell(cell_history)


@pytest.mark.parametrize(
    "field,value",
    [
        ("pid", 999),
        ("exit_code", 0),
        ("exit_observed", True),
        ("forced", True),
        ("signals", [15]),
        ("stdout_sha256", "0" * 64),
        ("stderr_sha256", "0" * 64),
        ("unknown", True),
    ],
)
def test_client_must_be_exact_never_launched_state(cell_history, field, value):
    pair = json.loads(cell_history.values["process-pair.json"])
    pair["client"][field] = value
    change_record(cell_history, "process-pair.json", **pair)
    with pytest.raises(ValueError):
        verify_cell(cell_history)


@pytest.mark.parametrize(
    "name,field,value",
    [
        ("service-intent.json", "command_sha256", "0" * 64),
        ("service-started.json", "pid", 999),
        ("service-process.json", "pid", 999),
        ("service-process.json", "exit_code", 1),
        ("service-process.json", "unknown", True),
        ("service-ready.json", "pid", 999),
        ("service-ready.json", "workload", "other"),
        ("service-ready.json", "host", "0.0.0.0"),
        ("service-ready.json", "port", True),
        ("service-ready.json", "port", 65536),
        ("service-ready.json", "unknown", True),
        ("service-cleanup.json", "pid", 999),
        ("service-cleanup.json", "status", "failed"),
        ("service-stop.json", "status", "unsent"),
        ("finalize.claim", "operation", "completion"),
        ("outcome.json", "status", "completion_prepared"),
        ("outcome.json", "stage", "completion"),
        ("outcome.json", "error_type", "exception"),
    ],
)
def test_lifecycle_and_finalization_records_join_exactly(
    cell_history, name, field, value
):
    change_record(cell_history, name, **{field: value})
    with pytest.raises(ValueError):
        verify_cell(cell_history)


@pytest.mark.parametrize("deadline", ["startup", "shutdown", "terminate", "kill"])
def test_no_changed_protective_deadline_accepts(cell_history, deadline):
    intent = json.loads(cell_history.values["process-pair-intent.json"])
    intent["deadlines"][deadline] += 1
    change_record(cell_history, "process-pair-intent.json", **intent)
    with pytest.raises(ValueError):
        verify_cell(cell_history)
