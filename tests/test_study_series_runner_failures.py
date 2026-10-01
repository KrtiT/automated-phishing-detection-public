"""Actual failed roots retain both accountings and never fabricate completed cells."""

import json
from pathlib import Path

import pytest
from study_series_runner_fixtures import candidates, manifests, run, series_case, setup
from test_study_series_runner import api

__all__ = ["candidates", "manifests", "series_case"]


def test_history_failure_precedes_every_output_reservation(
    tmp_path, series_case, monkeypatch
):
    case = setup(tmp_path, series_case, monkeypatch)

    def fail(public):
        raise ValueError("invented history verification failure")

    with pytest.raises(ValueError) as caught:
        run(case, hold_history=fail)
    assert api().series_run_failure(caught.value).series_attempt is None
    assert not (tmp_path / "series_attempt").exists()
    assert not (tmp_path / "cells_dir").exists()


def test_pre_cell_failure_retains_actual_prefix_and_both_failed_accountings(
    tmp_path, series_case, monkeypatch
):
    case = setup(tmp_path, series_case, monkeypatch)

    async def fail(*args, **kwargs):
        raise ValueError("invented pre-cell failure")

    monkeypatch.setattr(api(), "_run_series_cell", fail)
    with pytest.raises(ValueError) as caught:
        run(case)
    failure = api().series_run_failure(caught.value)
    assert failure.completed == () and failure.segment_candidate is None
    for role in ("series", "segment"):
        path = Path(case.source.profile["paths"][f"{role}_attempt"])
        accounting = json.loads((path / f"{role}-accounting.json").read_bytes())
        assert accounting["status"] == "failed"
        assert json.loads((path / "outcome.json").read_bytes())["status"] == "failed"
    assert failure.series_attempt is not None and failure.segment_attempt is not None
    assert case.events[-3:] == ["history_closed", "public_check", "supervisor_check"]


def test_failed_history_exit_cannot_turn_an_earlier_interrupt_into_success(
    tmp_path, series_case, monkeypatch
):
    from contextlib import contextmanager

    case = setup(tmp_path, series_case, monkeypatch)
    first = KeyboardInterrupt()

    @contextmanager
    def held(public):
        try:
            yield case.history
        finally:
            raise OSError("invented holder exit failure")

    async def fail(*args, **kwargs):
        raise first

    monkeypatch.setattr(api(), "_run_series_cell", fail)
    with pytest.raises(KeyboardInterrupt) as caught:
        run(case, hold_history=held)
    assert caught.value is first
    assert api().series_run_failure(caught.value).completed == ()
