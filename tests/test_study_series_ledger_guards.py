"""The live ledger neither reconstructs historical observations nor performs IO."""

import builtins
import os
import subprocess
from dataclasses import replace
from importlib import import_module
from pathlib import Path

import pytest
from study_series_ledger_fixtures import (
    api,
    candidate,
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
from study_series_prefix_fixtures import make_prefix_case
from test_study_series_ledger_ordering import rejection

from automated_phishing_detection._operational_process_records import ProcessObservation

__all__ = ["candidates", "manifests", "series_case", "prefix_case", "ledger_cell"]


def forbidden(*args, **kwargs):
    raise AssertionError("ledger crossed the IO/historical reconstruction boundary")


def test_acceptance_and_initialization_are_metadata_only(ledger_cell, monkeypatch):
    for owner, names in (
        (builtins, ("open",)),
        (os, ("open", "listdir", "scandir")),
        (Path, ("open", "read_bytes", "read_text", "stat", "lstat", "resolve")),
        (subprocess, ("run", "Popen")),
        (ProcessObservation, ("__init__",)),
    ):
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)
    for name in ("internal", "external", "cell"):
        module = import_module(f"automated_phishing_detection.study_history_{name}")
        monkeypatch.setattr(module, f"verify_historical_{name}_science", forbidden)
    current = ledger(ledger_cell.prefix)
    start(current, ledger_cell)
    for index, role in enumerate(("service", "client")):
        admission = current.issue(role, commands(ledger_cell)[index])
        admission.on_launched((321, 654)[index])
        admission.on_observed(True, 0)
    current.accept_cell(
        ledger_cell.candidate, ledger_cell.observation, ledger_cell.pair
    )
    assert snapshot(current)["cells"][0]["status"] == "accepted"


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(3)])
def test_interruption_preserved_and_issued_entry_retained(
    ledger_cell, monkeypatch, error
):
    from automated_phishing_detection import _study_series_ledger_issuance as issuance

    current = ledger(ledger_cell.prefix)
    start(current, ledger_cell)

    def interrupted(*args, **kwargs):
        raise error

    monkeypatch.setattr(issuance, "SeriesParentAdmission", interrupted)
    with pytest.raises(type(error)) as caught:
        current.issue("service", commands(ledger_cell)[0])
    assert caught.value is error
    current.stop_cell(api().SeriesCellStop("observation"))
    assert len(snapshot(current)["admissions"]) == 1


@pytest.mark.parametrize(
    "missing", ["service_launch", "client_launch", "service_exit", "client_exit"]
)
def test_successful_saved_process_bytes_cannot_replace_actual_callbacks(
    ledger_cell, missing
):
    current = ledger(ledger_cell.prefix)
    start(current, ledger_cell)
    service = current.issue("service", commands(ledger_cell)[0])
    if missing != "service_launch":
        service.on_launched(321)
        client = current.issue("client", commands(ledger_cell)[1])
        if missing != "client_launch":
            client.on_launched(654)
            if missing != "client_exit":
                client.on_observed(True, 0)
        if missing != "service_exit":
            service.on_observed(True, 0)
    with rejection():
        current.accept_cell(
            ledger_cell.candidate, ledger_cell.observation, ledger_cell.pair
        )


def test_single_last_cell_closes_exact_suffix_after_acceptance(series_case):
    import copy

    source = copy.copy(series_case)
    source.profile = copy.deepcopy(series_case.profile)
    source.profile["segment"]["start_ordinal"] = 125
    prefix = make_prefix_case(source)
    selected = prepared(prefix)
    science, observation, pair = candidate(selected)
    current = ledger(prefix)
    start(current, selected)
    observed(current, selected)
    current.accept_cell(science, observation, pair)
    value = snapshot(current)
    assert len(value["cells"]) == 1 and value["cells"][0]["ordinal"] == 125
    assert all(entry["accepted"] for entry in value["admissions"])
    with rejection():
        start(current, selected)


def test_stopped_candidate_wrong_inputs_cannot_be_smuggled(ledger_cell):
    current = ledger(ledger_cell.prefix)
    start(current, ledger_cell)
    candidate = replace(ledger_cell.candidate, reservation_sha256="0" * 64)
    with rejection():
        current.stop_cell(
            api().SeriesCellStop("finalization", candidate=candidate, publishing=True)
        )
