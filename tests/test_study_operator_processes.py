"""Actual v2 child admissions with invented authority, sources and model sessions."""

import json
import os
from base64 import b64decode
from dataclasses import replace

import pytest
import study_authorized_process_metadata_fixtures as metadata
import study_authorized_process_root_fixtures as roots
from study_authorized_process_cell_fixtures import run_first_cell
from study_operator_fixtures import POLICY_PATH, install_policy, seal
from test_source_runner import inputs, runner

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._study_admission import decode_admission_frame
from automated_phishing_detection._study_authorized_sources import run_adopted_sources
from automated_phishing_detection._study_child_commands import internal_command
from automated_phishing_detection.owned_worker import _observe_study_worker

__all__ = ["inputs", "runner"]


def _operator_setup(inputs, tmp_path, monkeypatch):
    original = roots.bind_metadata

    def bind_operator(case, patch):
        original(case, patch)
        install_policy(case, patch)
        (case.root / POLICY_PATH).write_bytes(case.policy)

    monkeypatch.setattr(roots, "bind_metadata", bind_operator)
    monkeypatch.setattr(metadata, "seal", lambda case, value: seal(value))
    return roots.setup(inputs, tmp_path, monkeypatch)


def test_actual_operator_admitted_sources_and_first_http_cell(
    tmp_path, inputs, monkeypatch
):
    case = _operator_setup(inputs, tmp_path, monkeypatch)
    envelope = json.loads(case.authorization.envelope_bytes)
    assert envelope["schema_version"] == 2
    assert "advisor" not in envelope["decisions"]
    assert envelope["technical_rebind"]["advisor_decision_obtained"] is False
    sources = run_adopted_sources(case.authorization, case.preparation, case.ledger)
    assert len(sources.internal.snapshot.payloads) == 35
    assert len(sources.external.snapshot.payloads) == 76
    completed = run_first_cell(case, sources, monkeypatch)
    assert len(completed.snapshot.run.warmup) == 1000
    assert len(completed.snapshot.run.measured) == 10000
    assert case.ledger.completed_cells == 1
    assert case.authorization.base.protected_evaluation_ready is False
    assert case.authorization.external.protected_evaluation_ready is False
    assert case.authorization.operational.protected_evaluation_ready is False
    _assert_owned_admissions(case)


def _assert_owned_admissions(case):
    entries = case.ledger.entries
    assert [entry["role"] for entry in entries] == [
        "internal",
        "external",
        "service",
        "client",
    ]
    for entry in entries:
        frame = decode_admission_frame(b64decode(entry["frame_bytes"], validate=True))
        assert frame.profile_sha256 == case.authorization.profile_sha256
        assert frame.envelope_sha256 == case.authorization.envelope_sha256
        assert frame.barrier_sha256 == case.ledger.barrier_sha256
        assert frame.parent_pid == os.getpid()
        assert entry["accepted"] and entry["exit_observed"] and entry["exit_code"] == 0
        with pytest.raises(ChildProcessError):
            os.waitpid(entry["launched_pid"], os.WNOHANG)


def _malformed_admission(case, change):
    envelope = json.loads(case.authorization.envelope_bytes)
    if change == "missing":
        del envelope["decisions"]["advisor_requirement_waiver"]
    else:
        envelope["decisions"]["advisor_requirement_waiver"]["operator"] = "Other person"
    content = canonical_bytes(envelope)
    case.envelope_path.write_bytes(content)
    malformed = replace(case.authorization, envelope_bytes=content)
    command = internal_command(malformed)
    admission = case.ledger.issue("internal", command)
    admission.frame = replace(
        admission.frame, envelope_sha256=malformed.envelope_sha256
    )
    return command, admission


def _watch_private_reads(case, tmp_path):
    marker = tmp_path / "invented-private-read-observed.txt"
    ready = tmp_path / "invented-private-read-watch-ready.txt"
    bootstrap = case.root.parent / "bootstrap/sitecustomize.py"
    watched = (
        "reservation.json",
        "study-intent.json",
        "prediction-barrier.json",
        "preparation-complete.json",
        "group_test.jsonl",
    )
    with bootstrap.open("a") as stream:
        stream.write(
            "\nimport sys\nfrom pathlib import Path\n"
            "def record_private_open(event, arguments):\n"
            f"    if event == 'open' and Path(str(arguments[0])).name in {watched!r}:\n"
            f"        Path({str(marker)!r}).write_text('private open observed')\n"
            "sys.addaudithook(record_private_open)\n"
            f"Path({str(ready)!r}).write_text('watch installed')\n"
        )
    return marker, ready


@pytest.mark.parametrize("change", ["missing", "misattributed"])
def test_actual_child_rejects_operator_waiver_before_private_access(
    tmp_path, inputs, monkeypatch, change
):
    case = _operator_setup(inputs, tmp_path, monkeypatch)
    marker, ready = _watch_private_reads(case, tmp_path)
    command, admission = _malformed_admission(case, change)
    observed = _observe_study_worker(command, admission)
    assert observed.exit.exit_observed and observed.exit.exit_code != 0
    assert case.ledger.entries[0]["launched_pid"] == observed.exit.pid
    assert case.ledger.entries[0]["accepted"] is False
    assert ready.is_file()
    assert not marker.exists()
    assert not case.authorization.paths.internal.attempt.exists()
    assert not case.authorization.paths.external.attempt.exists()
    with pytest.raises(ChildProcessError):
        os.waitpid(observed.exit.pid, os.WNOHANG)
