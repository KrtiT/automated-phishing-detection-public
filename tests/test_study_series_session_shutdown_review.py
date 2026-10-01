"""Shutdown budgets apply only after revocation, never to a running study."""

import json
import os
import signal
import subprocess
import time

import pytest
from study_series_session_fixtures import Process, api, observation
from test_study_series_session_cli import arguments, cli
from test_study_series_session_monitor import state
from test_study_series_session_process import setup


class Stubborn(Process):
    def wait(self, timeout=None):
        assert timeout is not None, "shutdown must have a protective budget"
        self.waits.append(timeout)
        if signal.SIGKILL not in self.signals:
            raise subprocess.TimeoutExpired("invented", timeout)
        self.returncode = -signal.SIGKILL
        return self.returncode

    def terminate(self):
        self.send_signal(signal.SIGTERM)

    def kill(self):
        self.send_signal(signal.SIGKILL)


def test_stubborn_root_is_escalated_only_after_revocation(monkeypatch):
    module, selected = api("_study_series_session_monitor"), state(Stubborn())
    groups = []

    def group_signal(process_id, number):
        assert selected.frames == ["revoked"]
        assert process_id == selected.process.pid
        if number == 0:
            raise ProcessLookupError
        groups.append(number)
        selected.process.send_signal(number)

    monkeypatch.setattr(os, "killpg", group_signal)
    module.stop_root(selected)
    assert selected.process.signals == [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]
    assert selected.process.waits == [180, 10, 10]
    assert groups == [signal.SIGTERM, signal.SIGKILL]
    assert selected.process.returncode == -signal.SIGKILL


def test_stubborn_inhibitor_is_reaped_and_actual_exit_recorded(monkeypatch):
    module, selected = api("_study_series_session_monitor"), state(Process(0))
    selected.inhibitor = Stubborn()
    monkeypatch.setattr(module.host, "command", lambda *unused: "no owned assertions")
    module.stop_inhibitor(selected)
    assert selected.inhibitor.waits == [10, 10]
    assert selected.recorded["sleep-cleanup.json"]["exit_code"] == -signal.SIGKILL


def test_root_launch_exception_keeps_post_and_inhibitor_cleanup(monkeypatch, tmp_path):
    module, path, children = setup(monkeypatch, tmp_path)
    original = module.subprocess.Popen

    def spawn(command, **keywords):
        if command[0] != "/usr/bin/caffeinate":
            raise OSError("synthetic root launch failure")
        return original(command, **keywords)

    monkeypatch.setattr(module.subprocess, "Popen", spawn)
    with pytest.raises(OSError, match="synthetic root launch failure"):
        module.run_series_session(cli()._parser().parse_args(arguments()))
    assert len(children) == 1 and children[0].poll() is not None
    assert (path / "post.json").stat().st_size > 0
    assert (path / "sleep-cleanup.json").stat().st_size > 0


def test_root_owns_an_isolated_process_group(monkeypatch, tmp_path):
    module, unused_path, children = setup(monkeypatch, tmp_path, running=True)

    def observe(selected):
        assert os.getpgid(selected.process.pid) == selected.process.pid
        module.monitor.stop_root(selected)

    monkeypatch.setattr(module.monitor, "monitor", observe)
    monkeypatch.setattr(module.host, "capture", observation)
    assert module.run_series_session(cli()._parser().parse_args(arguments())) != 0
    assert all(child.poll() is not None for child in children)


def test_real_stubborn_root_is_killed_and_cleanup_evidence_survives(
    monkeypatch, tmp_path
):
    module, path, children = setup(monkeypatch, tmp_path, running=True)
    script = tmp_path / "synthetic_root.py"
    script.write_text(
        script.read_text().replace(
            "import sys, time",
            "import sys, time, signal\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
        )
    )
    monkeypatch.setattr(
        module.monitor,
        "DEADLINES",
        {
            "shutdown": 0.1,
            "terminate": 0.1,
            "kill": 1,
        },
    )

    def interrupt_ready(selected):
        deadline = time.monotonic() + 10
        while b"admitted_synthetic_root" not in (path / "stdout.log").read_bytes():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        raise KeyboardInterrupt("synthetic stop after actual admission")

    monkeypatch.setattr(module.monitor, "monitor", interrupt_ready)
    with pytest.raises(KeyboardInterrupt):
        module.run_series_session(cli()._parser().parse_args(arguments()))
    assert children[1].returncode == -signal.SIGKILL
    assert all(child.poll() is not None for child in children)
    post = json.loads((path / "post.json").read_bytes())
    assert post["root_shutdown_escalated"] is True
    assert post["root_exit_code"] == -signal.SIGKILL
    assert (path / "sleep-cleanup.json").stat().st_size > 0
