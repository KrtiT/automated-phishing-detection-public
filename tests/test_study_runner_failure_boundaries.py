"""Root accounting retains actual inner stages and never fabricates continuation."""

import json
from types import SimpleNamespace

import pytest
import study_runner_fixtures as fixtures
from study_run_record_fixtures import capacity, prepared
from study_runner_matrix_fixtures import install

from automated_phishing_detection._operational_cell_results import (
    VerifiedOperationalCell,
    VerifiedOperationalWorking,
)
from automated_phishing_detection._operational_process_records import ProcessObservation
from automated_phishing_detection.internal_process_handoff import (
    ObservedInternalCompletion,
)
from automated_phishing_detection.operational_cell_inputs import RestoredOperationalCell
from automated_phishing_detection.operational_cell_runner import (
    ObservedOperationalCell,
    OperationalCellFailure,
)
from automated_phishing_detection.operational_schedule import cell_for_ordinal

__all__ = ["prepared"]


def test_original_inner_cell_failure_stage_remains_available(
    tmp_path, prepared, monkeypatch
):
    case = fixtures.setup(tmp_path, prepared, monkeypatch, capacity(prepared))
    install(case, monkeypatch)
    original = ValueError("inner rejected")
    original.operational_failure = OperationalCellFailure(
        cell_for_ordinal(1), None, None, None, None, "completion", False
    )
    original.progress = b"original process progress"

    async def fail(*arguments, **keywords):
        raise original

    monkeypatch.setattr(case.body, "_run_bound_cell", fail)
    with pytest.raises(case.module.StudyRunError) as caught:
        fixtures.execute(case)
    retained = caught.value.study_failure
    assert retained.cells is not None
    assert retained.cells[0].stopped.failure is original.operational_failure
    assert retained.cells[0].stopped.stage == "completion"
    accounting = json.loads((case.paths.attempt / "study-accounting.json").read_bytes())
    assert accounting["cells"][0]["stage"] == "completion"


def returned_cell():
    inputs = RestoredOperationalCell(b"", b"", b"", b"", cell_for_ordinal(1), ())
    working = VerifiedOperationalWorking((), inputs, None, b"{}\n", "a" * 64)
    return ObservedOperationalCell(
        ProcessObservation(b"invented actual-return marker"),
        VerifiedOperationalCell((), working),
    )


def test_post_return_packing_failure_preserves_accepted_unpacked_cell(
    tmp_path, prepared, monkeypatch
):
    case = fixtures.setup(tmp_path, prepared, monkeypatch, capacity(prepared))
    install(case, monkeypatch)
    returned = returned_cell()

    async def complete(*arguments, **keywords):
        case.events.append("one_cell")
        return returned

    def fail(*arguments, **keywords):
        raise ValueError("packing failed")

    monkeypatch.setattr(case.body, "_run_bound_cell", complete)
    monkeypatch.setattr(case.body, "retain_accepted_cell", fail)
    with pytest.raises(case.module.StudyRunError) as caught:
        fixtures.execute(case)
    slots = caught.value.study_failure.cells
    assert slots[0].returned is returned and slots[0].status == "accepted"
    assert all(slot.status == "unattempted" for slot in slots[1:])
    assert case.events.count("one_cell") == 1 and "reduce" not in case.events


@pytest.mark.parametrize("internal_accepted", [False, True])
def test_source_pair_failure_keeps_exact_source_progress(
    tmp_path, prepared, monkeypatch, internal_accepted
):
    case = fixtures.setup(tmp_path, prepared, monkeypatch, capacity(prepared))
    original = ValueError("source rejected")
    if internal_accepted:
        original.source_internal = ObservedInternalCompletion(None, None)
        original.external_failure = SimpleNamespace(stage="external_observation")

    def fail(*arguments):
        raise original

    monkeypatch.setattr(case.body, "_run_observed_prepared_sources", fail)
    with pytest.raises(case.module.StudyRunError) as caught:
        fixtures.execute(case)
    assert all(
        slot.status == "unattempted" for slot in caught.value.study_failure.cells
    )
    if internal_accepted:
        assert caught.value.source_internal is original.source_internal
        assert caught.value.external_failure is original.external_failure
    accounting = json.loads((case.paths.attempt / "study-accounting.json").read_bytes())
    assert (accounting["internal_status"], accounting["external_status"]) == (
        ("accepted", "stopped") if internal_accepted else ("stopped", "unattempted")
    )


def test_planned_but_unattempted_injected_leaf_prevents_hold_publication(
    tmp_path, prepared, monkeypatch
):
    case = fixtures.setup(tmp_path, prepared, monkeypatch)
    original = case.body._run_bound_preparation

    def injected(*arguments):
        (case.paths.cells_directory / "cell-125-attempt").mkdir(mode=0o700)
        return original(*arguments)

    monkeypatch.setattr(case.body, "_run_bound_preparation", injected)
    with pytest.raises(case.module.StudyRunError):
        fixtures.execute(case)
    assert not case.paths.public_summary.exists()
    assert not (case.paths.attempt / "prediction-barrier.json").exists()
