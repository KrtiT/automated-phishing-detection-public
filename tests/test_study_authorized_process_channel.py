"""Official child rejection of copied admission bytes and changed root metadata."""

import os
import subprocess

import pytest
from study_authorized_process_root_fixtures import setup
from test_source_runner import inputs, runner

from automated_phishing_detection._study_authorized_sources import run_adopted_sources
from automated_phishing_detection._study_child_commands import internal_command
from automated_phishing_detection.source_runner import SourceExecutionError

__all__ = ["inputs", "runner"]


def test_copied_frame_file_cannot_admit_official_child(tmp_path, inputs, monkeypatch):
    case = setup(inputs, tmp_path, monkeypatch)
    command = internal_command(case.authorization)
    admission = case.ledger.issue("internal", command)
    copied = tmp_path / "copied-frame.json"
    copied.write_bytes(admission.frame.canonical_bytes)
    with copied.open("rb") as stream:
        environment = dict(os.environ, APD_STUDY_ADMISSION_FD=str(stream.fileno()))
        result = subprocess.run(
            command,
            env=environment,
            pass_fds=(stream.fileno(),),
            capture_output=True,
            text=True,
            check=False,
        )
    assert result.returncode != 0
    assert not case.authorization.paths.internal.attempt.exists()
    assert case.ledger.entries[0]["launched_pid"] is None


def test_changed_retained_barrier_rejected_before_source_attempt(
    tmp_path, inputs, monkeypatch
):
    case = setup(inputs, tmp_path, monkeypatch)
    barrier = case.authorization.paths.attempt / "prediction-barrier.json"
    barrier.write_bytes(
        barrier.read_bytes().replace(b"necessary_capacity_present", b"whole_study_hold")
    )
    with pytest.raises(SourceExecutionError, match="worker_exit"):
        run_adopted_sources(case.authorization, case.preparation, case.ledger)
    assert len(case.ledger.entries) == 1
    assert case.ledger.entries[0]["exit_observed"] is True
    assert case.ledger.entries[0]["accepted"] is False
    assert not case.authorization.paths.internal.attempt.exists()
