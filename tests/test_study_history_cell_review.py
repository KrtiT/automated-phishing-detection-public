"""Independent science review excludes fabricated live ownership and IO."""

import builtins
import os
import subprocess
from pathlib import Path

import pytest
from study_history_cell_fixtures import (
    api,
    candidates,
    case,
    history,
    manifests,
    restore,
)
from test_stopped_study_authorization_review import _guard_observations

from automated_phishing_detection import _study_history_cell_science as science
from automated_phishing_detection._operational_cell_results import (
    VerifiedOperationalCell,
    VerifiedOperationalWorking,
)
from automated_phishing_detection.execution_preflight import ExecutionBinding
from automated_phishing_detection.operational_inputs import AcceptedOperationalInputs
from automated_phishing_detection.owned_worker import WorkerObservation

__all__ = ["candidates", "case", "history", "manifests"]


def forbidden(*args, **kwargs):
    pytest.fail("historical arithmetic used IO or constructed live ownership")


def test_no_io_or_fabricated_process_and_acceptance(history, monkeypatch):
    _guard_observations(monkeypatch)
    for record in (
        ExecutionBinding,
        WorkerObservation,
        AcceptedOperationalInputs,
        VerifiedOperationalWorking,
        VerifiedOperationalCell,
    ):
        monkeypatch.setattr(record, "__init__", forbidden)
    for module, name in (
        (builtins, "open"),
        (os, "open"),
        (Path, "open"),
        (subprocess, "Popen"),
    ):
        monkeypatch.setattr(module, name, forbidden)
    result = restore(history)
    assert result.summary_bytes == history.working.summary_bytes
    assert not hasattr(result, "accepted")


@pytest.mark.parametrize(
    "name", ["attempt/run.json", "attempt/measured.json", "public-summary.json"]
)
def test_unmatched_bytes_cannot_reach_numerical_kernels(history, monkeypatch, name):
    monkeypatch.setattr(science, "summary", forbidden)
    values = history.values | {name: history.values[name] + b" "}
    with pytest.raises(api().HistoricalCellScienceError):
        restore(history, payloads=tuple(values.items()))


def counted(original, calls, name):
    def call(*args, **kwargs):
        calls.append(name)
        return original(*args, **kwargs)

    return call


def test_original_numerical_kernels_are_used_without_live_completer(
    history, monkeypatch
):
    calls = []
    names = (
        "decode_run",
        "verify_shift_checkpoints",
        "verify_http_checkpoints",
        "verify_offline_trace",
        "summarize_shift_run",
        "summarize_run",
    )
    for name in names:
        monkeypatch.setattr(science, name, counted(getattr(science, name), calls, name))
    restore(history)
    assert calls == (
        [
            "decode_run",
            "verify_shift_checkpoints",
            "verify_offline_trace",
            "summarize_shift_run",
        ]
        if history.working.inputs.cell.workload == "shift_period"
        else ["decode_run", "verify_http_checkpoints", "summarize_run"]
    )
