"""Actual two-child loopback execution with invented rows and synthetic scores."""

import json
import os

import pytest
from operational_cell_runner_children import write_children
from operational_cell_runner_fixtures import execute, setup
from operational_input_fixtures import candidates, manifests

from automated_phishing_detection._operational_cell_protocol import SNAPSHOT_NAMES

__all__ = ["candidates", "manifests"]


def assert_exits(record):
    value = json.loads(record)
    for role in ("service", "client"):
        assert value[role]["exit_observed"] is True
        assert value[role]["forced"] is False
        with pytest.raises(ChildProcessError):
            os.waitpid(value[role]["pid"], os.WNOHANG)
    return value


@pytest.mark.parametrize("ordinal", [21, 121])
def test_actual_children_complete_http_and_shift_saved_acceptance(
    tmp_path, manifests, monkeypatch, ordinal
):
    case = setup(tmp_path, manifests, monkeypatch, ordinal, tmp_path / "checkout")
    write_children(case)
    result = execute(case)
    observed = assert_exits(result.observation.record)
    assert observed["service"]["exit_code"] == observed["client"]["exit_code"] == 0
    assert set(dict(result.snapshot.payloads)) == set(SNAPSHOT_NAMES)
    assert result.snapshot.accepted is case.accepted
    run = result.snapshot.run
    assert len(run.warmup) == 1000
    assert len(run.measured) == (10000 if ordinal == 21 else 1001)
    assert run.after_measured.completed_requests == len(run.warmup) + len(run.measured)
    assert (
        json.loads(case.paths.public_summary.read_bytes())["status"]
        == "operational_evidence_published"
    )


def test_actual_client_nonzero_after_run_retention_never_publishes(
    tmp_path, manifests, monkeypatch
):
    case = setup(tmp_path, manifests, monkeypatch, 121, tmp_path / "checkout")
    write_children(case, mode="client_nonzero")
    with pytest.raises(case.module.OperationalCellExecutionError) as caught:
        execute(case)
    progress = assert_exits(caught.value.progress)
    assert progress["client"]["exit_code"] == 17
    assert (case.paths.attempt / "run.json").exists()
    assert not case.paths.public_summary.exists()
    assert caught.value.operational_failure.observation is None


def test_actual_zero_exits_do_not_accept_a_forged_saved_trace(
    tmp_path, manifests, monkeypatch
):
    case = setup(tmp_path, manifests, monkeypatch, 121, tmp_path / "checkout")
    write_children(case, mode="corrupt_trace")
    with pytest.raises(case.module.OperationalCellExecutionError) as caught:
        execute(case)
    failure = caught.value.operational_failure
    observed = assert_exits(failure.observation.record)
    assert observed["client"]["exit_code"] == observed["service"]["exit_code"] == 0
    assert failure.working is failure.candidate is None
    assert not failure.publishing and not case.paths.public_summary.exists()
