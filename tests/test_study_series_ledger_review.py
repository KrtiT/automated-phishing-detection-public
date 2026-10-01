"""Independent checks of actual observation use and interrupted acceptance."""

import pytest
from study_series_ledger_fixtures import (
    api,
    candidates,
    ledger,
    ledger_cell,
    manifests,
    observed,
    prefix_case,
    series_case,
    snapshot,
    start,
)

from automated_phishing_detection import _study_series_ledger_acceptance as acceptance
from automated_phishing_detection import _study_series_ledger_records as records

__all__ = ["candidates", "manifests", "series_case", "prefix_case", "ledger_cell"]


def ready(selected):
    current = ledger(selected.prefix)
    start(current, selected)
    observed(current, selected)
    return current


def test_unchanged_process_verifier_receives_actual_observation(
    ledger_cell, monkeypatch
):
    verifier = acceptance.process.verify_process_records
    seen = []

    def checked(*arguments, **keywords):
        seen.append(keywords["observation"])
        return verifier(*arguments, **keywords)

    monkeypatch.setattr(acceptance.process, "verify_process_records", checked)
    current = ready(ledger_cell)
    current.accept_cell(
        ledger_cell.candidate, ledger_cell.observation, ledger_cell.pair
    )
    assert len(seen) == 1 and seen[0] is ledger_cell.observation
    assert snapshot(current)["cells"][0]["holders_closed"] is True


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(4)])
def test_interrupted_acceptance_cannot_leave_half_accepted_pair(
    ledger_cell, monkeypatch, error
):
    current = ready(ledger_cell)

    def interrupted(*arguments, **keywords):
        raise error

    monkeypatch.setattr(records, "completed", interrupted)
    with pytest.raises(type(error)) as caught:
        current.accept_cell(
            ledger_cell.candidate, ledger_cell.observation, ledger_cell.pair
        )
    assert caught.value is error
    current.stop_cell(api().SeriesCellStop("finalization"))
    saved = snapshot(current)
    assert saved["cells"][0]["status"] == "stopped"
    assert all(not entry["accepted"] for entry in saved["admissions"])


def test_snapshots_are_detached_from_later_consumer_mutation(ledger_cell):
    current = ready(ledger_cell)
    current.accept_cell(
        ledger_cell.candidate, ledger_cell.observation, ledger_cell.pair
    )
    previous = api().snapshot_series_ledger(current)
    decoded = snapshot(current)
    decoded["admissions"][0]["accepted"] = False
    decoded["cells"][0]["snapshot_sha256"].clear()
    assert api().snapshot_series_ledger(current) == previous
