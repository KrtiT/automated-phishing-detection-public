"""Fixture-only tests of the bounded live parent and complete-study barrier."""

import hashlib
import json
import os
import threading

import pytest
import revision_admission as admission


def make_parent(tmp_path, readiness=None):
    source = tmp_path / "source"
    source.write_bytes(b"frozen")
    profile = {
        "files": {str(source): hashlib.sha256(b"frozen").hexdigest()},
        "readiness": readiness or dict.fromkeys(admission.TASKS, True),
    }
    return admission.LiveParent(profile, tmp_path / "attempt", lambda: None), source


def test_whole_study_readiness_rejects_missing_task_before_work(tmp_path):
    parent, unused = make_parent(tmp_path, {"P": True, "F": True, "R": True})
    with pytest.raises(admission.AdmissionError, match="whole_study_not_ready"):
        parent.start()
    assert not (parent.root / "parent-started.json").exists()


def test_exact_order_success_and_closed_callback(tmp_path):
    parent, unused = make_parent(tmp_path)
    parent.start()
    with pytest.raises(admission.AdmissionError, match="task_order"):
        parent.admit("R")
    for name in admission.TASKS:
        callback = parent.admit(name)
        callback()
        parent.complete(name)
        with pytest.raises(admission.AdmissionError, match="task_not_live"):
            callback()
    parent.finish()
    assert json.loads((parent.root / "parent-completed.json").read_bytes())[
        "completed"
    ] == list(admission.TASKS)


def test_input_mutation_revokes_live_callback_and_completion(tmp_path):
    parent, source = make_parent(tmp_path)
    parent.start()
    callback = parent.admit("P")
    source.write_bytes(b"changed")
    with pytest.raises(admission.AdmissionError, match="bound_file_changed"):
        callback()
    with pytest.raises(admission.AdmissionError):
        parent.complete("P")


def test_same_length_restored_mtime_mutation_detected(tmp_path):
    parent, source = make_parent(tmp_path)
    parent.start()
    callback = parent.admit("P")
    observed = source.stat()
    source.write_bytes(b"alter!")
    os.utime(source, ns=(observed.st_atime_ns, observed.st_mtime_ns))
    with pytest.raises(admission.AdmissionError, match="bound_file_changed"):
        callback()


def test_parent_failure_retains_unstarted_and_never_restarts(tmp_path):
    parent, unused = make_parent(tmp_path)
    parent.start()
    parent.admit("P")
    parent.stop(ValueError("fixture"))
    record = json.loads((parent.root / "parent-failure.json").read_bytes())
    assert record["active"] == "P" and record["unstarted"] == ["F", "R", "C"]
    assert not (parent.root / "parent-completed.json").exists()
    with pytest.raises(admission.AdmissionError):
        parent.start()


def test_foreign_thread_cannot_borrow_admission(tmp_path):
    parent, unused = make_parent(tmp_path)
    parent.start()
    callback = parent.admit("P")
    errors = []

    def invoke():
        try:
            callback()
        except admission.AdmissionError as error:
            errors.append(str(error))

    child = threading.Thread(target=invoke)
    child.start()
    child.join()
    assert errors == ["parent_owner_changed"]


def test_thermal_warning_and_unknown_state_fail_closed():
    for value in (
        "Thermal_Level = 1",
        "",
        "No thermal warning level has been recorded",
    ):
        with pytest.raises(admission.AdmissionError, match="thermal_state_not_normal"):
            admission.verify_thermal(value)
    admission.verify_thermal(
        "No thermal warning level has been recorded\nNo performance warning level has been recorded\nNo CPU power status has been recorded"
    )


def test_stale_or_revoked_monitor_cannot_admit_work():
    guard = admission.MonitorState()
    with pytest.raises(admission.AdmissionError, match="session_observation_stale"):
        guard.check()
    guard.observe()
    guard.check()
    guard.revoke("ac_power_absent")
    guard.observe()
    with pytest.raises(admission.AdmissionError, match="ac_power_absent"):
        guard.check()


def test_code_guard_rechecked_at_task_boundary_before_work(tmp_path):
    parent, unused = make_parent(tmp_path)
    changed = []

    def code_check():
        if changed:
            raise admission.AdmissionError("code_changed")

    parent.code_check = code_check
    parent.start()
    changed.append(True)
    with pytest.raises(admission.AdmissionError, match="code_changed"):
        parent.admit("P")
    assert parent.active is None


def test_host_session_holds_before_yield_and_cleans_inhibitor(tmp_path, monkeypatch):
    class Inhibitor:
        pid = 123
        stopped = False

        def terminate(self):
            self.stopped = True

        def wait(self, timeout):
            return 0

    inhibitor = Inhibitor()
    monkeypatch.setattr(admission.subprocess, "Popen", lambda *args: inhibitor)
    monkeypatch.setattr(admission.time, "sleep", lambda *args: None)
    monkeypatch.setattr(admission.host, "capture", lambda: {"fixture": True})
    monkeypatch.setattr(
        admission.host, "violation", lambda observation: "ac_power_absent"
    )
    with pytest.raises(admission.AdmissionError, match="ac_power_absent"):
        with admission.guarded_session(tmp_path / "host.jsonl", stable_seconds=0):
            pytest.fail("unsafe session yielded")
    assert inhibitor.stopped
    assert json.loads((tmp_path / "host.jsonl").read_bytes()) == {"fixture": True}
