"""Rehashed forged receipts cannot be accepted merely because their pins match."""

import json

import pytest
from operational_input_fixtures import candidates, manifests
from stopped_study_authorization_fixtures import verify as verify_root
from study_history_snapshot_fixtures import complete_history, verify
from study_history_snapshot_mutation_fixtures import (
    move_source,
    repin_source,
    replace_cell,
    replace_source_record,
    republish_cell,
)
from study_history_snapshot_source_fixtures import encoded
from study_run_record_fixtures import prepared

__all__ = ["candidates", "manifests", "prepared"]


@pytest.mark.parametrize("role", ["internal", "external"])
@pytest.mark.parametrize(
    "name,field,value",
    [
        ("attempt/finalize.claim", "operation", "failure"),
        ("attempt/outcome.json", "status", "failed"),
        ("attempt/outcome.json", "private_sha256", {}),
        ("public-summary.json", "private_sha256", {}),
        ("public-summary.json", "status", "whole_study_hold"),
        ("public-summary.json", "checkpoint_sha256", {}),
    ],
)
def test_rehashed_source_receipt_corruption_rejects(
    prepared, manifests, role, name, field, value
):
    case = complete_history(prepared, manifests)
    replace_source_record(
        case, role, name, lambda record: record.update({field: value})
    )
    verify_root(case)
    with pytest.raises(ValueError):
        verify(case)


@pytest.mark.parametrize("role", ["internal", "external"])
def test_rehashed_source_copy_disagreement_rejects(prepared, manifests, role):
    case = complete_history(prepared, manifests)
    getattr(case, role)["attempt/evidence/bindings.json"] = b"different opaque bytes"
    repin_source(case, role)
    verify_root(case)
    with pytest.raises(ValueError):
        verify(case)


@pytest.mark.parametrize("role", ["internal", "external"])
def test_noncanonical_source_receipt_rejects_after_repin(prepared, manifests, role):
    case = complete_history(prepared, manifests)
    getattr(case, role)["attempt/finalize.claim"] += b"\n"
    repin_source(case, role)
    verify_root(case)
    with pytest.raises(ValueError):
        verify(case)


@pytest.mark.parametrize(
    "name", ["attempt/finalize.claim", "attempt/outcome.json", "public-summary.json"]
)
def test_rehashed_cell_receipt_corruption_rejects(prepared, manifests, name):
    case = complete_history(prepared, manifests)
    value = json.loads(dict(case.cells[0][2])[name])
    value["unexpected"] = True
    replace_cell(case, name, encoded(value))
    verify_root(case)
    with pytest.raises(ValueError):
        verify(case)


@pytest.mark.parametrize(
    "name",
    [
        "attempt/evidence/run.json",
        "attempt/service-started.json",
        "attempt/client-role.json",
    ],
)
def test_rehashed_cell_copies_and_process_records_reject(prepared, manifests, name):
    case = complete_history(prepared, manifests)
    replace_cell(case, name, b"different opaque bytes")
    verify_root(case)
    with pytest.raises(ValueError):
        verify(case)


@pytest.mark.parametrize("role", ["internal", "external"])
def test_fully_rehashed_source_path_substitution_rejects(prepared, manifests, role):
    case = complete_history(prepared, manifests)
    move_source(case, role)
    verify_root(case)
    with pytest.raises(ValueError):
        verify(case)


def test_fully_rehashed_cell_path_substitution_rejects(prepared, manifests):
    from stopped_study_authorization_fixtures import refresh_accounting
    from study_history_snapshot_cell_fixtures import cells

    case = complete_history(prepared, manifests)
    case.profile["paths"]["cells-dir"] = "/invented/unapproved-cells"
    cells(case)
    refresh_accounting(case)
    verify_root(case)
    with pytest.raises(ValueError):
        verify(case)


@pytest.mark.parametrize("name", ["service-started.json", "client-role.json"])
def test_republished_process_lies_reject_after_all_copies_and_hashes_match(
    prepared, manifests, name
):
    case = complete_history(prepared, manifests)
    value = json.loads(dict(case.cells[0][2])[f"attempt/{name}"])
    value["pid"] += 1
    republish_cell(case, name, value)
    verify_root(case)
    with pytest.raises(ValueError):
        verify(case)
