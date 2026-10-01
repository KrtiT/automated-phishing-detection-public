"""Late holder failures and interruptions cannot promote a fresh candidate."""

import json
from contextlib import contextmanager

import pytest
from study_series_cell_runner_fixtures import (
    candidates,
    execute,
    inputs,
    install_observer,
    manifests,
    series_case,
    setup,
)
from study_series_ledger_fixtures import snapshot
from test_study_series_cell_runner import api

__all__ = ["candidates", "manifests", "series_case"]


def test_late_public_recheck_keeps_complete_candidate_stopped(
    tmp_path, series_case, monkeypatch
):
    case = setup(tmp_path, series_case, monkeypatch)
    install_observer(case, monkeypatch)
    calls = []

    def recheck(public):
        calls.append(public)
        if len(calls) == 2:
            raise ValueError("invented final recheck failure")

    monkeypatch.setattr(api(), "recheck_series_public_execution", recheck)
    with inputs(case), pytest.raises(ValueError):
        execute(case)
    saved = snapshot(case.ledger)
    stopped = saved["cells"][0]
    assert stopped["status"] == "stopped" and stopped["holders_closed"] is False
    assert stopped["publishing"] is True and len(stopped["snapshot_sha256"]) == 36
    assert not any(entry["accepted"] for entry in saved["admissions"])


def test_acceptance_follows_successful_completer_exit(
    tmp_path, series_case, monkeypatch
):
    case = setup(tmp_path, series_case, monkeypatch)
    install_observer(case, monkeypatch)
    hold, accept = api().hold_series_cell, case.ledger.accept_cell

    @contextmanager
    def completion(*arguments, **keywords):
        with hold(*arguments, **keywords) as held:
            case.completer = held
            yield held
        case.events.append("completion_closed")

    def accepted(*arguments):
        assert case.completer._closed is True
        assert case.events[-1] == "completion_closed"
        case.events.append("accepted")
        return accept(*arguments)

    monkeypatch.setattr(api(), "hold_series_cell", completion)
    monkeypatch.setattr(case.ledger, "accept_cell", accepted)
    with inputs(case):
        execute(case)
    assert case.events == ["observe", "completion_closed", "accepted"]


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), SystemExit(7)])
def test_observer_interruption_survives_later_writer_error(
    tmp_path, series_case, monkeypatch, failure
):
    case = setup(tmp_path, series_case, monkeypatch)
    writer = api().held_attempt_writer

    @contextmanager
    def broken_writer(*arguments, **keywords):
        with writer(*arguments, **keywords) as held:
            try:
                yield held
            finally:
                raise OSError("invented writer exit failure")

    async def interrupted(*arguments, **keywords):
        raise failure

    monkeypatch.setattr(api(), "held_attempt_writer", broken_writer)
    monkeypatch.setattr(api(), "observe_series_operational_children", interrupted)
    with inputs(case), pytest.raises(type(failure)) as caught:
        execute(case)
    assert caught.value is failure
    assert snapshot(case.ledger)["cells"][0]["status"] == "stopped"


@pytest.mark.parametrize("failure", [ValueError("invented"), KeyboardInterrupt()])
def test_pre_observation_failure_retains_actual_reservation_and_inputs(
    tmp_path, series_case, monkeypatch, failure
):
    case = setup(tmp_path, series_case, monkeypatch)

    def interrupted(*arguments, **keywords):
        raise failure

    monkeypatch.setattr(api(), "hold_series_cell", interrupted)
    with inputs(case), pytest.raises(type(failure)):
        execute(case)
    saved = snapshot(case.ledger)
    assert saved["admissions"] == []
    assert (
        saved["cells"][0]["reservation_sha256"]
        == case.selected.attempt.reservation_sha256
    )
    outcome = json.loads(
        (case.selected.attempt.directory / "outcome.json").read_bytes()
    )
    assert outcome["status"] == "failed"
