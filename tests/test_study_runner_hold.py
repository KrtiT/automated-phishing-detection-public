"""Whole-study holds precede scoring; branch fixtures are not source authority."""

import json
from dataclasses import replace

import pytest
import study_runner_fixtures as fixtures
from study_run_record_fixtures import REQUIREMENTS, isolated_shortage, prepared

__all__ = ["prepared"]


def assert_hold(case, result):
    public = json.loads(result.snapshot.payload("public-summary.json"))
    assert public["status"] == "whole_study_hold"
    assert public["feasibility"] == case.retained.feasibility
    assert len(result.snapshot.payloads) == 10
    assert len(result.cells) == 125
    assert all(slot.status == "unattempted" for slot in result.cells)
    assert result.sources is result.accepted is None
    assert result.preparation is case.retained
    assert case.events.count("prepare") == case.events.count("hold_preparation") == 1
    assert case.events.count("release_preparation") == 1
    assert list(case.paths.cells_directory.iterdir()) == []
    assert not case.paths.internal.attempt.exists()
    assert not case.paths.external.attempt.exists()
    assert not case.paths.accepted_inputs_directory.exists()


def test_genuine_in_memory_preparation_shortages_hold_every_task(
    tmp_path, prepared, monkeypatch
):
    case = fixtures.setup(tmp_path, prepared, monkeypatch)
    assert_hold(case, fixtures.execute(case))


@pytest.mark.parametrize("requirement", REQUIREMENTS)
def test_each_promised_task_shortage_holds_the_whole_study(
    tmp_path, prepared, monkeypatch, requirement
):
    case = fixtures.setup(
        tmp_path, prepared, monkeypatch, isolated_shortage(prepared, requirement)
    )
    assert_hold(case, fixtures.execute(case))
    barrier = json.loads((case.paths.attempt / "prediction-barrier.json").read_bytes())
    assert barrier["predictions_started"] is False
    assert barrier["feasibility"]["shortages"][0]["requirement"] == requirement


@pytest.mark.parametrize("changed", ["buffer", "reservation", "completion"])
def test_fresh_preparation_must_equal_original_held_bytes_before_barrier(
    tmp_path, prepared, monkeypatch, changed
):
    case = fixtures.setup(tmp_path, prepared, monkeypatch)
    if changed == "buffer":
        case.fresh = replace(
            case.fresh,
            payloads=case.fresh.payloads[:-1]
            + (("preparation-complete.json", b"{}\n"),),
        )
    elif changed == "reservation":
        case.fresh = replace(case.fresh, reservation_sha256="0" * 64)
    else:
        case.retained = replace(case.retained, completion_sha256="0" * 64)
    with pytest.raises(case.module.StudyRunError):
        fixtures.execute(case)
    assert not (case.paths.attempt / "prediction-barrier.json").exists()
    assert not case.paths.public_summary.exists()


def test_preparation_failure_retains_all_unattempted_cells(
    tmp_path, prepared, monkeypatch
):
    case = fixtures.setup(tmp_path, prepared, monkeypatch)
    original = ValueError("invented private detail")
    original.preparation_progress = b"original retained progress"

    def fail(*arguments):
        raise original

    monkeypatch.setattr(case.body, "_run_bound_preparation", fail)
    with pytest.raises(case.module.StudyRunError) as caught:
        fixtures.execute(case)
    assert caught.value.preparation_progress == original.preparation_progress
    assert all(
        slot.status == "unattempted" for slot in caught.value.study_failure.cells
    )
    assert "private detail" not in str(caught.value)
    accounting = json.loads((case.paths.attempt / "study-accounting.json").read_bytes())
    assert accounting["status"] == "failed"
    assert (
        accounting["internal_status"] == accounting["external_status"] == "unattempted"
    )
    assert not case.paths.public_summary.exists()
