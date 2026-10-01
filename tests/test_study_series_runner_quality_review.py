"""Independent privacy and post-acceptance interruption boundary checks."""

from types import SimpleNamespace

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
from study_series_ledger_fixtures import commands, snapshot

from automated_phishing_detection._study_series_cell_runner_state import (
    SeriesCellFailure,
)

__all__ = ["candidates", "manifests", "series_case"]


@pytest.mark.parametrize(
    "name", ("cell", "attempt", "observation", "working", "candidate")
)
def test_series_failure_repr_omits_private_context_like_original(name):
    values = dict.fromkeys(("cell", "attempt", "observation", "working", "candidate"))
    values[name] = SimpleNamespace(private="invented_private_failure_context")
    failure = SeriesCellFailure(**values, stage="observation", publishing=False)
    assert "invented_private_failure_context" not in repr(failure)
    assert "stage='observation'" in repr(failure)


@pytest.mark.parametrize("interruption", (KeyboardInterrupt(), SystemExit(9)))
def test_interrupt_after_atomic_acceptance_preserves_acceptance_and_stops_issuance(
    tmp_path, series_case, monkeypatch, interruption
):
    case = setup(tmp_path, series_case, monkeypatch)
    install_observer(case, monkeypatch)
    accept = case.ledger.accept_cell

    def interrupted(*arguments):
        accept(*arguments)
        raise interruption

    monkeypatch.setattr(case.ledger, "accept_cell", interrupted)
    with inputs(case), pytest.raises(type(interruption)) as caught:
        execute(case)
    saved = snapshot(case.ledger)
    assert caught.value is interruption
    assert saved["cells"][0]["status"] == "accepted"
    assert all(entry["accepted"] for entry in saved["admissions"])
    assert all(cell["status"] == "unattempted" for cell in saved["cells"][1:])
    assert caught.value.operational_failure.candidate is not None
    with pytest.raises(ValueError):
        case.ledger.issue("service", commands(case.selected)[0])
