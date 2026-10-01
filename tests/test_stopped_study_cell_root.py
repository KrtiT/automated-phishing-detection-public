"""Whole-root reauthentication and exactly one stopped service admission remain mandatory."""

import copy

import pytest
from operational_input_fixtures import candidates, manifests
from stopped_study_authorization_fixtures import change_frame, refresh_accounting
from stopped_study_cell_fixtures import cell_history, invented_cell_history, verify_cell
from study_run_record_fixtures import prepared

__all__ = [
    "candidates",
    "manifests",
    "prepared",
    "cell_history",
    "invented_cell_history",
]


@pytest.mark.parametrize("count", [0, 2])
def test_only_one_stopped_service_admission_accepts(cell_history, count):
    root = cell_history.root
    entries = root.accounting["authorization_ledger"]["admissions"]
    if count == 0:
        entries.pop()
    else:
        client = copy.deepcopy(entries[-1])
        client.update(role="client", command_sha256="b" * 64)
        entries.append(client)
        change_frame(root, -1, role="client", command_sha256="b" * 64)
    refresh_accounting(root)
    with pytest.raises(ValueError):
        verify_cell(cell_history)


@pytest.mark.parametrize(
    "field,value",
    [
        ("stage", "completion"),
        ("publishing", True),
        ("publishing", None),
        ("observation_sha256", "0" * 64),
        ("reservation_sha256", "0" * 64),
        ("progress_sha256", "0" * 64),
        ("progress_sha256", None),
    ],
)
def test_exact_original_failure_projection_is_required(cell_history, field, value):
    root = cell_history.root
    root.scientific["cells"][cell_history.ordinal - 1][field] = value
    refresh_accounting(root)
    with pytest.raises(ValueError):
        verify_cell(cell_history)


@pytest.mark.parametrize(
    "field,value",
    [
        ("launched_pid", 999),
        ("launched_pid", None),
        ("exit_observed", False),
        ("exit_code", 1),
        ("accepted", True),
    ],
)
def test_original_service_admission_must_match_clean_process(
    cell_history, field, value
):
    root = cell_history.root
    root.accounting["authorization_ledger"]["admissions"][-1][field] = value
    refresh_accounting(root)
    with pytest.raises(ValueError):
        verify_cell(cell_history)


@pytest.mark.parametrize(
    "field",
    [
        "command_sha256",
        "cell_binding_sha256",
        "root_reservation_sha256",
        "accepted_inputs_sha256",
        "profile_sha256",
    ],
)
def test_rehashed_admission_keeps_every_original_identity(cell_history, field):
    root = cell_history.root
    if field == "command_sha256":
        root.accounting["authorization_ledger"]["admissions"][-1][field] = "0" * 64
    change_frame(root, -1, **{field: "0" * 64})
    with pytest.raises(ValueError):
        verify_cell(cell_history)
