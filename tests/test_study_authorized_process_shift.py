"""Actual shift children; the 120-cell scheduling prefix is explicitly simulated."""

import json
import os

import pytest
from study_authorized_process_shift_fixtures import run_shift_cell, setup_shift
from test_source_runner import inputs, runner

from automated_phishing_detection._study_authorized_sources import run_adopted_sources

__all__ = ["inputs", "runner"]


def test_actual_admitted_shift_pair_after_explicit_simulated_prefix(
    tmp_path, inputs, monkeypatch
):
    case = setup_shift(inputs, tmp_path, monkeypatch)
    sources = run_adopted_sources(case.authorization, case.preparation, case.ledger)
    completion = run_shift_cell(case, sources, monkeypatch)
    observed = json.loads(completion.observation.record)
    assert len(completion.snapshot.run.warmup) == 1000
    assert len(completion.snapshot.run.measured) == 1000
    assert len(completion.snapshot.run.trace.rows) == 1000
    assert len(case.ledger.entries[2:-2]) == 240
    assert all("accepted" not in entry for entry in case.ledger.entries[2:-2])
    assert len(case.ledger.cell_acceptances) == 1
    assert case.ledger.cell_acceptances[0]["cell_ordinal"] == 121
    for entry in case.ledger.entries[-2:]:
        assert entry["cell_ordinal"] == 121 and entry["accepted"]
        assert entry["exit_observed"] and entry["exit_code"] == 0
        assert observed[entry["role"]]["pid"] == entry["launched_pid"]
        assert observed[entry["role"]]["forced"] is False
        with pytest.raises(ChildProcessError):
            os.waitpid(entry["launched_pid"], os.WNOHANG)
