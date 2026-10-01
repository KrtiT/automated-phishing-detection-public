"""Issuance and actual callback failures cannot open a replacement suffix."""

import os

import pytest
from study_series_ledger_fixtures import (
    api,
    candidates,
    commands,
    ledger,
    ledger_cell,
    manifests,
    observed,
    prefix_case,
    prepared,
    series_case,
    snapshot,
    start,
)

__all__ = ["candidates", "manifests", "series_case", "prefix_case", "ledger_cell"]


def rejection():
    return pytest.raises(api().SeriesLedgerError, match="^invalid_series_ledger$")


def test_next_cell_cannot_start_before_current_acceptance(ledger_cell):
    current = ledger(ledger_cell.prefix)
    start(current, ledger_cell)
    with rejection():
        start(current, ledger_cell)
    with rejection():
        current.issue("service", commands(ledger_cell)[0])


def test_client_requires_service_actual_launch_callback(ledger_cell):
    current = ledger(ledger_cell.prefix)
    start(current, ledger_cell)
    current.issue("service", commands(ledger_cell)[0])
    with rejection():
        current.issue("client", commands(ledger_cell)[1])
    current.stop_cell(api().SeriesCellStop("observation"))
    assert len(snapshot(current)["admissions"]) == 1


@pytest.mark.parametrize("role", ["service", "client"])
def test_duplicate_role_issuance_is_permanently_closed(ledger_cell, role):
    current = ledger(ledger_cell.prefix)
    start(current, ledger_cell)
    admission = current.issue("service", commands(ledger_cell)[0])
    admission.on_launched(321)
    if role == "client":
        current.issue("client", commands(ledger_cell)[1])
    with rejection():
        current.issue(role, commands(ledger_cell)[role == "client"])
    current.stop_cell(api().SeriesCellStop("observation"))
    assert len(snapshot(current)["admissions"]) == (2 if role == "client" else 1)


@pytest.mark.parametrize("pid", [None, True, 0, -1, 3.0, "321", "parent"])
def test_invalid_callback_pid_prevents_later_issuance(ledger_cell, pid):
    current = ledger(ledger_cell.prefix)
    start(current, ledger_cell)
    admission = current.issue("service", commands(ledger_cell)[0])
    with rejection():
        admission.on_launched(os.getpid() if pid == "parent" else pid)
    with rejection():
        current.issue("client", commands(ledger_cell)[1])


@pytest.mark.parametrize(
    "failure",
    ["duplicate_launch", "early_exit", "duplicate_exit", "same_pid", "invalid_exit"],
)
def test_callback_order_and_pair_pid_alias_reject(ledger_cell, failure):
    current = ledger(ledger_cell.prefix)
    start(current, ledger_cell)
    first = current.issue("service", commands(ledger_cell)[0])
    if failure != "early_exit":
        first.on_launched(321)
    if failure == "duplicate_exit":
        first.on_observed(True, 0)
    if failure == "same_pid":
        second = current.issue("client", commands(ledger_cell)[1])
    actions = {
        "duplicate_launch": lambda: first.on_launched(321),
        "early_exit": lambda: first.on_observed(True, 0),
        "duplicate_exit": lambda: first.on_observed(True, 0),
        "same_pid": lambda: second.on_launched(321),
        "invalid_exit": lambda: first.on_observed(True, True),
    }
    with rejection():
        actions[failure]()
    with rejection():
        current.accept_cell(
            ledger_cell.candidate, ledger_cell.observation, ledger_cell.pair
        )


@pytest.mark.parametrize("seen,code", [(False, None), (True, 1), (True, -15)])
def test_bad_exit_is_retained_and_later_cleanup_callback_survives(
    ledger_cell, seen, code
):
    current = ledger(ledger_cell.prefix)
    start(current, ledger_cell)
    first = current.issue("service", commands(ledger_cell)[0])
    first.on_launched(321)
    second = current.issue("client", commands(ledger_cell)[1])
    second.on_launched(654)
    second.on_observed(seen, code)
    first.on_observed(True, 0)
    current.stop_cell(api().SeriesCellStop("observation"))
    entries = snapshot(current)["admissions"]
    assert entries[0]["exit_code"] == 0
    assert entries[1]["observation_recorded"] is True
    assert entries[1]["exit_observed"] is seen and entries[1]["exit_code"] == code
    with rejection():
        current.accept_cell(
            ledger_cell.candidate, ledger_cell.observation, ledger_cell.pair
        )


def test_pipe_constructor_failure_keeps_preappended_entry(ledger_cell, monkeypatch):
    from automated_phishing_detection import _study_series_ledger_issuance as issuance

    current = ledger(ledger_cell.prefix)
    start(current, ledger_cell)
    monkeypatch.setattr(
        issuance,
        "SeriesParentAdmission",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("private")),
    )
    with rejection():
        current.issue("service", commands(ledger_cell)[0])
    current.stop_cell(api().SeriesCellStop("observation"))
    entry = snapshot(current)["admissions"][0]
    assert entry["issued"] is True and entry["launched_pid"] is None


def test_pid_reuse_after_previous_cell_acceptance_is_allowed(ledger_cell):
    current = ledger(ledger_cell.prefix)
    start(current, ledger_cell)
    observed(current, ledger_cell)
    current.accept_cell(
        ledger_cell.candidate, ledger_cell.observation, ledger_cell.pair
    )
    second = prepared(ledger_cell.prefix, ledger_cell.cell.ordinal + 1)
    start(current, second)
    observed(current, second)
    current.stop_cell(api().SeriesCellStop("completion"))
    assert [entry["launched_pid"] for entry in snapshot(current)["admissions"]] == [
        321,
        654,
        321,
        654,
    ]


@pytest.mark.parametrize(
    "operation", ["start", "issue", "callback", "stop", "snapshot"]
)
def test_every_operation_stays_in_actual_initial_parent(
    ledger_cell, monkeypatch, operation
):
    current = ledger(ledger_cell.prefix)
    if operation != "start":
        start(current, ledger_cell)
    admission = (
        current.issue("service", commands(ledger_cell)[0])
        if operation == "callback"
        else None
    )
    actions = {
        "start": lambda: start(current, ledger_cell),
        "issue": lambda: current.issue("service", commands(ledger_cell)[0]),
        "callback": lambda: admission.on_launched(321),
        "stop": lambda: current.stop_cell(api().SeriesCellStop("observation")),
        "snapshot": lambda: api().snapshot_series_ledger(current),
    }
    monkeypatch.setattr(os, "getpid", lambda: 123456789)
    with rejection():
        actions[operation]()
