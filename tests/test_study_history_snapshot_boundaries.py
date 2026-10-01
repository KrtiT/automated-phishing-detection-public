"""Historical byte consistency is complete-prefix only and grants no live role."""

import builtins
import os
import subprocess
from pathlib import Path

import pytest
from operational_input_fixtures import candidates, manifests
from study_history_snapshot_fixtures import arguments, complete_history, verify
from study_run_record_fixtures import prepared

from automated_phishing_detection import (
    _operational_process_records,
    _owned_process_exit,
    external_source_completion,
    internal_process_handoff,
    operational_inputs,
    owned_worker,
    saved_evidence,
)

__all__ = ["candidates", "manifests", "prepared"]


@pytest.mark.parametrize(
    "change",
    ["missing", "duplicate", "reordered", "wrong_ordinal", "wrong_reservation"],
)
def test_complete_prefix_is_required(prepared, manifests, change):
    case = complete_history(prepared, manifests, prefix=2)
    first, second = case.cells
    altered = {
        "missing": (first,),
        "duplicate": (first, first),
        "reordered": (second, first),
        "wrong_ordinal": ((2, first[1], first[2]), second),
        "wrong_reservation": ((1, "0" * 64, first[2]), second),
    }[change]
    with pytest.raises(ValueError):
        verify(case, cell_snapshots=altered)


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "reordered", "false_ordinal"]
)
def test_independent_cell_manifests_are_ordered_and_exact(prepared, manifests, change):
    case = complete_history(prepared, manifests, prefix=2)
    first, second = arguments(case)["expected_cells_sha256"]
    altered = {
        "missing": (first,),
        "duplicate": (first, first),
        "reordered": (second, first),
        "false_ordinal": ((True, first[1]), second),
    }[change]
    with pytest.raises(ValueError):
        verify(case, expected_cells_sha256=altered)


def forbidden(*args, **kwargs):
    pytest.fail("history verification attempted IO, replay or a live observation")


def test_verifier_uses_neither_io_replay_nor_observed_objects(
    prepared, manifests, monkeypatch
):
    case = complete_history(prepared, manifests)
    targets = [
        (builtins, "open"),
        (os, "open"),
        (Path, "open"),
        (subprocess, "Popen"),
        (_operational_process_records, "ProcessObservation"),
        (_owned_process_exit, "OwnedProcessExit"),
        (owned_worker, "WorkerObservation"),
        (internal_process_handoff, "ObservedInternalCompletion"),
        (operational_inputs, "AcceptedOperationalInputs"),
        (operational_inputs, "build_accepted_inputs"),
        (saved_evidence, "reconstruct_internal_evidence"),
        (external_source_completion, "reconstruct_external_evidence"),
    ]
    for module, name in targets:
        monkeypatch.setattr(module, name, forbidden)
    result = verify(case)
    assert result.authorization.accepted_ordinals == (1,)
    assert "opaque science" not in repr(result)
