"""Real synthetic subprocesses exercise supervision without any research access."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from study_series_session_fixtures import api, observation
from test_study_series_session_cli import arguments, cli


def setup(monkeypatch, tmp_path, *, running=False):
    module = api("study_series_session")
    tmp_path.chmod(0o700)
    destination = tmp_path / "physical"
    public = SimpleNamespace(
        base=SimpleNamespace(root=Path(__file__).parents[1], revision="a" * 40),
        profile_sha256="b" * 64,
        envelope_sha256="c" * 64,
        profile_bytes=json.dumps(
            {"paths": {"physical_records_dir": str(destination)}}
        ).encode(),
    )
    script = tmp_path / "synthetic_root.py"
    script.write_text(
        "\n".join(
            [
                "import sys, time",
                "from types import SimpleNamespace",
                "from automated_phishing_detection.study_series_supervision import consume_series_supervision",
                "arguments = SimpleNamespace(expected_revision='a'*40, expected_profile_sha256='b'*64, expected_envelope_sha256='c'*64)",
                "with consume_series_supervision(arguments) as held:",
                "    held.check()",
                "    print('admitted_synthetic_root', flush=True)",
                "    time.sleep(120)" if running else "    held.check()",
            ]
        )
    )
    monkeypatch.setattr(
        module, "bind_series_public_execution", lambda *args, **kwargs: public
    )
    monkeypatch.setattr(module, "require_final_policy", lambda selected: None)
    monkeypatch.setattr(
        module, "recheck_series_public_execution", lambda selected: None
    )
    monkeypatch.setattr(module.host, "capture", observation)
    monkeypatch.setattr(
        module.host, "command", lambda *args: "no synthetic inhibitor remains"
    )
    monkeypatch.setattr(
        module.host,
        "inhibition",
        lambda sample, process: (
            None if process.poll() is None else "owned_sleep_inhibitor_exited"
        ),
    )
    monkeypatch.setattr(
        module, "root_command", lambda selected: [sys.executable, str(script)]
    )
    original, children = subprocess.Popen, []

    def spawn(command, **keywords):
        if command[0] == "/usr/bin/caffeinate":
            command = [sys.executable, "-c", "import time; time.sleep(120)"]
        child = original(command, **keywords)
        children.append(child)
        return child

    monkeypatch.setattr(module.subprocess, "Popen", spawn)
    return module, destination, children


def test_real_root_consumes_live_parent_frame_once_and_all_children_exit(
    monkeypatch, tmp_path
):
    module, path, children = setup(monkeypatch, tmp_path)
    assert module.run_series_session(cli()._parser().parse_args(arguments())) == 0
    assert len(children) == 2 and all(child.poll() is not None for child in children)
    assert (path / "stdout.log").read_bytes() == b"admitted_synthetic_root\n"
    launch = json.loads((path / "launch.json").read_bytes())
    assert (
        launch["supervisor_pid"] == os.getpid()
        and launch["root_pid"] == children[1].pid
    )
    assert json.loads((path / "post.json").read_bytes())["root_exit_code"] == 0
    assert (
        json.loads((path / "sleep-cleanup.json").read_bytes())["exit_code"] is not None
    )


def test_interrupt_preserves_same_exception_stops_root_and_retains_actual_exit(
    monkeypatch, tmp_path
):
    module, path, children = setup(monkeypatch, tmp_path, running=True)
    interrupted = KeyboardInterrupt("synthetic operator interruption")

    def interrupt(selected):
        raise interrupted

    monkeypatch.setattr(module.monitor, "monitor", interrupt)
    with pytest.raises(KeyboardInterrupt) as caught:
        module.run_series_session(cli()._parser().parse_args(arguments()))
    assert caught.value is interrupted
    assert len(children) == 2 and all(child.poll() is not None for child in children)
    post = json.loads((path / "post.json").read_bytes())
    assert post["root_exit_code"] != 0
    assert post["supervisor_error_type"] == "KeyboardInterrupt"
    assert post["session_violation"] == "supervisor_interrupted"


def test_failed_release_does_not_retry_or_leave_a_waiting_root(monkeypatch, tmp_path):
    module, path, children = setup(monkeypatch, tmp_path)

    def reject(self, frame):
        raise ValueError("synthetic failed release")

    channel = api("_study_series_supervisor_channel")
    monkeypatch.setattr(channel.SupervisorChannel, "release", reject)
    with pytest.raises(ValueError, match="synthetic failed release"):
        module.run_series_session(cli()._parser().parse_args(arguments()))
    assert len(children) == 2 and all(child.poll() is not None for child in children)
    assert b"admitted" not in (path / "stdout.log").read_bytes()
    assert json.loads((path / "post.json").read_bytes())["root_exit_code"] != 0


def test_prelaunch_failure_preserves_records_and_reaps_inhibitor(monkeypatch, tmp_path):
    module, path, children = setup(monkeypatch, tmp_path)

    def reject(public):
        raise ValueError("synthetic changed binding")

    monkeypatch.setattr(module, "recheck_series_public_execution", reject)
    with pytest.raises(ValueError, match="synthetic changed binding"):
        module.run_series_session(cli()._parser().parse_args(arguments()))
    assert len(children) == 1 and children[0].poll() is not None
    assert json.loads((path / "post.json").read_bytes())["root_pid"] is None
    assert (path / "launch.json").read_bytes() == b""
