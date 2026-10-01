"""Adverse conditions revoke one root and remain visible after process cleanup."""

import io
import signal
from types import SimpleNamespace

import pytest
from study_series_session_fixtures import Process, api, inhibitor, observation


def state(process=None):
    recorded, frames = {}, []
    return SimpleNamespace(
        process=Process() if process is None else process,
        inhibitor=inhibitor(),
        records=SimpleNamespace(
            record=lambda name, value: recorded.setdefault(name, value),
            streams={"conditions.jsonl": io.BytesIO()},
            check=lambda: None,
        ),
        channel=SimpleNamespace(
            revoke=lambda: frames.append("revoked"),
            heartbeat=lambda stamp: frames.append(stamp),
        ),
        selected=None,
        failure=None,
        recorded=recorded,
        frames=frames,
    )


def test_clean_monitor_sends_heartbeat_and_preserves_post(monkeypatch):
    module, selected = api("_study_series_session_monitor"), state()
    monkeypatch.setattr(module.host, "capture", observation)
    module.monitor(selected)
    module.post(selected)
    assert len(selected.frames) == 1 and type(selected.frames[0]) is int
    assert selected.process.waits == [30]
    assert selected.recorded["post.json"]["root_exit_code"] == 0
    assert selected.selected is None


@pytest.mark.parametrize("violation", ["battery", "inhibitor", "workload"])
def test_bad_sample_revokes_before_interrupt_and_never_heartbeats(
    monkeypatch, violation
):
    module, selected = api("_study_series_session_monitor"), state()
    sample = observation()
    if violation == "battery":
        sample["battery"] = "Now drawing from 'Battery Power'"
    elif violation == "inhibitor":
        selected.inhibitor = inhibitor(1)
    else:
        sample["competing_known_workload_pids"] = [42]
    monkeypatch.setattr(module.host, "capture", lambda: sample)
    module.monitor(selected)
    module.post(selected)
    assert selected.frames == ["revoked"]
    assert selected.process.signals == [signal.SIGINT]
    assert selected.recorded["post.json"]["session_violation"] is not None
    assert selected.recorded["post.json"]["root_exit_code"] == 130


def test_postcondition_failure_cannot_be_reported_as_success(monkeypatch):
    module, selected = api("_study_series_session_monitor"), state(Process(0))
    monkeypatch.setattr(
        module.host, "capture", lambda: observation(battery="Battery Power")
    )
    module.post(selected)
    assert selected.selected == "ac_power_absent"
    assert selected.recorded["post.json"]["root_exit_code"] == 0


def test_unavailable_postconditions_preserve_actual_exit(monkeypatch):
    module, selected = api("_study_series_session_monitor"), state(Process(130))

    def broken():
        raise RuntimeError("private detail must not enter record")

    monkeypatch.setattr(module.host, "capture", broken)
    module.post(selected)
    assert selected.selected == "postconditions_unavailable"
    assert selected.recorded["post.json"]["root_exit_code"] == 130
    assert "private detail" not in repr(selected.recorded)


def test_wait_timeout_is_only_a_sampling_interval(monkeypatch):
    module, selected = api("_study_series_session_monitor"), state()
    monkeypatch.setattr(module.host, "capture", observation)
    attempts = []

    def wait(timeout=None):
        attempts.append(timeout)
        if len(attempts) < 4:
            raise module.subprocess.TimeoutExpired("invented", timeout)
        selected.process.returncode = 0

    selected.process.wait = wait
    module.monitor(selected)
    assert attempts == [30] * 4
    assert selected.process.signals == []
    assert len(selected.frames) == 4


def test_ready_waits_for_owned_assertions(monkeypatch):
    module = api("_study_series_session_monitor")
    samples, waits = [observation(assertions=""), observation()], []
    monkeypatch.setattr(module.host, "capture", lambda: samples.pop(0))
    monkeypatch.setattr(module.time, "sleep", waits.append)
    sample, observed_ns = module.ready(inhibitor())
    assert sample == observation() and observed_ns > 0
    assert waits == [0.1] and samples == []


def test_root_exit_between_sample_and_heartbeat_is_not_false_failure(monkeypatch):
    module, selected = api("_study_series_session_monitor"), state()
    monkeypatch.setattr(module.host, "capture", observation)

    def exited(observed_ns):
        selected.process.returncode = 0
        raise ValueError("series_supervision_delivery_failed")

    selected.channel.heartbeat = exited
    module.monitor(selected)
    module.post(selected)
    assert selected.selected is None
    assert selected.recorded["post.json"]["root_exit_code"] == 0


def test_live_root_heartbeat_failure_is_not_swallowed(monkeypatch):
    module, selected = api("_study_series_session_monitor"), state()
    monkeypatch.setattr(module.host, "capture", observation)

    def broken(observed_ns):
        raise ValueError("series_supervision_delivery_failed")

    selected.channel.heartbeat = broken
    with pytest.raises(ValueError, match="series_supervision_delivery_failed"):
        module.monitor(selected)
