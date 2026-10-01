"""Retained admission inventories and source observations cannot be substituted."""

import pytest
from operational_input_fixtures import candidates, manifests
from stopped_study_authorization_fixtures import (
    change_frame,
    make_stopped,
    refresh_accounting,
    verify,
)
from study_run_record_fixtures import prepared

__all__ = ["candidates", "manifests", "prepared"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("unknown", True),
        ("schema_version", True),
        ("protocol", "other"),
        ("intent_sha256", "0" * 64),
        ("barrier_sha256", "0" * 64),
        ("handoff_sha256", "0" * 64),
        ("source_results_sha256", "0" * 64),
        ("accepted_inputs_sha256", "0" * 64),
    ],
)
def test_ledger_headers_join_original_records(prepared, manifests, field, value):
    case = make_stopped(prepared, manifests)
    case.accounting["authorization_ledger"][field] = value
    refresh_accounting(case)
    with pytest.raises(ValueError):
        verify(case)


@pytest.mark.parametrize(
    "change",
    [
        "missing_admission",
        "later_admission",
        "missing_acceptance",
        "extra_acceptance",
        "duplicate_acceptance",
        "wrong_acceptance_ordinal",
        "unknown_acceptance",
        "non_list_admissions",
        "non_list_acceptances",
    ],
)
def test_all_and_only_prefix_admissions_are_retained(prepared, manifests, change):
    case = make_stopped(prepared, manifests, prefix=2, stopped_admissions=2)
    ledger = case.accounting["authorization_ledger"]
    if change == "missing_admission":
        ledger["admissions"].pop(0)
    elif change == "later_admission":
        ledger["admissions"].append(ledger["admissions"][-1])
    elif change == "missing_acceptance":
        ledger["cell_acceptances"].pop()
    elif change == "extra_acceptance":
        ledger["cell_acceptances"].append(ledger["cell_acceptances"][-1])
    elif change == "duplicate_acceptance":
        ledger["cell_acceptances"][1] = ledger["cell_acceptances"][0]
    elif change == "wrong_acceptance_ordinal":
        ledger["cell_acceptances"][0]["cell_ordinal"] = True
    elif change == "unknown_acceptance":
        ledger["cell_acceptances"][0]["extra"] = True
    else:
        ledger[change.removeprefix("non_list_")] = {}
    refresh_accounting(case)
    with pytest.raises(ValueError):
        verify(case)


@pytest.mark.parametrize("position", [0, 1, 2, 3])
@pytest.mark.parametrize(
    "field,value",
    [
        ("launched_pid", 999),
        ("exit_observed", False),
        ("exit_code", 1),
        ("accepted", False),
        ("issued", False),
        ("extra", True),
    ],
)
def test_original_source_and_prefix_observations_match(
    prepared, manifests, position, field, value
):
    case = make_stopped(prepared, manifests)
    case.accounting["authorization_ledger"]["admissions"][position][field] = value
    refresh_accounting(case)
    with pytest.raises(ValueError):
        verify(case)


@pytest.mark.parametrize("position", [0, 1, 2, 3])
def test_rehashed_command_frames_must_match_original_processes(
    prepared, manifests, position
):
    case = make_stopped(prepared, manifests)
    entry = case.accounting["authorization_ledger"]["admissions"][position]
    entry["command_sha256"] = "0" * 64
    change_frame(case, position, command_sha256="0" * 64)
    with pytest.raises(ValueError):
        verify(case)
