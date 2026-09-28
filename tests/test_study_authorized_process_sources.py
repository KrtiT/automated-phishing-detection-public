"""Actual official child scripts under real profile, pipe and retained-root gates."""

import json
import os
from base64 import b64decode
from dataclasses import replace
from hashlib import sha256

import pytest
from study_authorized_process_cell_fixtures import run_first_cell
from study_authorized_process_root_fixtures import setup
from test_source_runner import inputs, runner

from automated_phishing_detection._study_admission import decode_admission_frame
from automated_phishing_detection._study_authorized_sources import run_adopted_sources
from automated_phishing_detection.source_runner import SourceExecutionError

__all__ = ["inputs", "runner"]


def test_actual_study_children_accept_sources_then_first_http_cell(
    tmp_path, inputs, monkeypatch
):
    case = setup(inputs, tmp_path, monkeypatch)
    result = run_adopted_sources(case.authorization, case.preparation, case.ledger)
    entries = json.loads(case.ledger.snapshot())["admissions"]
    assert [entry["role"] for entry in entries] == ["internal", "external"]
    assert all(
        entry["accepted"] and entry["exit_observed"] and entry["exit_code"] == 0
        for entry in entries
    )
    assert len(result.internal.snapshot.payloads) == 35
    assert len(result.external.snapshot.payloads) == 76
    assert result.internal.worker.exit.pid != result.external.worker.exit.pid
    cell = run_first_cell(case, result, monkeypatch)
    observed = json.loads(cell.observation.record)
    assert observed["service"]["exit_code"] == observed["client"]["exit_code"] == 0
    assert len(cell.snapshot.run.warmup) == 1000
    assert len(cell.snapshot.run.measured) == 10000
    assert case.ledger.completed_cells == 1
    _assert_admissions(case)


def _assert_admissions(case):
    entries = case.ledger.entries
    assert [entry["role"] for entry in entries] == [
        "internal",
        "external",
        "service",
        "client",
    ]
    for entry in entries:
        content = b64decode(entry["frame_bytes"], validate=True)
        assert sha256(content).hexdigest() == entry["frame_sha256"]
        frame = decode_admission_frame(content)
        assert frame.parent_pid == os.getpid()
        assert frame.profile_sha256 == case.authorization.profile_sha256
        assert frame.envelope_sha256 == case.authorization.envelope_sha256
        assert frame.barrier_sha256 == case.ledger.barrier_sha256
        assert entry["accepted"] and entry["exit_observed"] and entry["exit_code"] == 0
        with pytest.raises(ChildProcessError):
            os.waitpid(entry["launched_pid"], os.WNOHANG)


def test_published_internal_child_nonzero_exit_never_admits_external(
    tmp_path, inputs, monkeypatch
):
    case = setup(inputs, tmp_path, monkeypatch, "nonzero")
    with pytest.raises(SourceExecutionError, match="worker_exit"):
        run_adopted_sources(case.authorization, case.preparation, case.ledger)
    entries = json.loads(case.ledger.snapshot())["admissions"]
    assert len(entries) == 1 and entries[0]["role"] == "internal"
    assert entries[0]["exit_observed"] is True and entries[0]["exit_code"] == 17
    assert entries[0]["accepted"] is False
    assert case.authorization.paths.internal.public_summary.is_file()
    assert not case.authorization.paths.external.attempt.exists()


def test_forged_profile_frame_rejected_by_actual_child_before_source_attempt(
    tmp_path, inputs, monkeypatch
):
    case = setup(inputs, tmp_path, monkeypatch)
    original = case.ledger.issue

    def forged(*args, **kwargs):
        admission = original(*args, **kwargs)
        admission.frame = replace(admission.frame, profile_sha256="f" * 64)
        return admission

    monkeypatch.setattr(case.ledger, "issue", forged)
    with pytest.raises(SourceExecutionError, match="worker_exit"):
        run_adopted_sources(case.authorization, case.preparation, case.ledger)
    assert case.ledger.entries[0]["exit_observed"] is True
    assert case.ledger.entries[0]["accepted"] is False
    assert not case.authorization.paths.internal.attempt.exists()
    assert not case.authorization.paths.external.attempt.exists()
