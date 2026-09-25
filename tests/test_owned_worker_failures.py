import json
import os
import signal
import subprocess

import pytest
from test_owned_worker import _command, _module


@pytest.fixture
def owned_resources(monkeypatch):
    module = _module()
    launch, temporary, kill = subprocess.Popen, module.tempfile.TemporaryFile, os.kill
    processes, streams = [], []

    def start(*args, **kwargs):
        process = launch(*args, **kwargs)
        processes.append(process)
        return process

    def create(*args, **kwargs):
        stream = temporary(*args, **kwargs)
        streams.append(stream)
        return stream

    monkeypatch.setattr(module.subprocess, "Popen", start)
    monkeypatch.setattr(module.tempfile, "TemporaryFile", create)
    yield module, processes, streams
    for process in processes:
        try:
            waited_pid, status = os.waitpid(process.pid, os.WNOHANG)
            if not waited_pid:
                kill(process.pid, signal.SIGKILL)
                os.waitpid(process.pid, 0)
        except (ProcessLookupError, ChildProcessError):
            pass
        process.returncode = -signal.SIGKILL
    for stream in streams:
        stream.close()


def test_launch_failure_is_symbolic_and_closes_streams(monkeypatch, owned_resources):
    module, processes, streams = owned_resources

    def fail(*args, **kwargs):
        raise OSError("private command path")

    monkeypatch.setattr(module.subprocess, "Popen", fail)
    with pytest.raises(
        module.WorkerExecutionError, match="^worker_launch_failed$"
    ) as caught:
        module.observe_worker(_command("raise SystemExit(0)"))
    progress = json.loads(caught.value.progress)
    assert processes == []
    assert progress["pid"] is None
    assert progress["exit_observed"] is False
    assert progress["failure"] == "worker_launch_failed"
    assert "private" not in str(caught.value) + caught.value.progress.decode()
    assert len(streams) == 2 and all(stream.closed for stream in streams)


def test_wait_failure_joins_only_owned_child(monkeypatch, owned_resources):
    module, processes, streams = owned_resources
    original = module.observe_owned_exit
    waits = []

    def fail_once(process, *, block=False):
        waits.append((process.pid, block))
        if len(waits) == 1:
            raise OSError("private wait diagnostic")
        return original(process, block=block)

    monkeypatch.setattr(module, "observe_owned_exit", fail_once)
    with pytest.raises(
        module.WorkerExecutionError, match="^worker_wait_failed$"
    ) as caught:
        module.observe_worker(_command("import time; time.sleep(60)"))
    progress = json.loads(caught.value.progress)
    assert len(processes) == 1
    assert progress["pid"] == processes[0].pid
    assert progress["exit_observed"] is True
    assert progress["exit_code"] == -signal.SIGKILL
    assert progress["signals"] == [signal.SIGKILL]
    assert all(stream.closed for stream in streams)
    with pytest.raises(ChildProcessError):
        os.waitpid(processes[0].pid, os.WNOHANG)


@pytest.mark.parametrize("lost", [False, True])
def test_diagnostic_failure_keeps_terminal_cache(monkeypatch, owned_resources, lost):
    module, processes, streams = owned_resources
    original = module.observe_owned_exit
    observations = []

    def observe(process, *, block=False):
        observations.append(block)
        if lost:
            os.waitpid(process.pid, 0)
        return original(process, block=block)

    def fail(stream):
        raise OSError("private diagnostic path")

    def forbidden(*args, **kwargs):
        pytest.fail("terminal observation must never be signaled")

    monkeypatch.setattr(module, "observe_owned_exit", observe)
    monkeypatch.setattr(module, "_stream_hash", fail)
    monkeypatch.setattr(module.os, "kill", forbidden)
    with pytest.raises(
        module.WorkerExecutionError, match="^worker_diagnostics_failed$"
    ) as caught:
        module.observe_worker(_command("raise SystemExit(17)"))
    progress = json.loads(caught.value.progress)
    assert observations == [True]
    assert progress["exit_observed"] is not lost
    assert progress["exit_code"] == (None if lost else 17)
    assert all(stream.closed for stream in streams)


@pytest.mark.parametrize("position", [0, 1])
def test_setup_failure_closes_already_allocated_streams(
    monkeypatch, owned_resources, position
):
    module, processes, streams = owned_resources
    original = module.tempfile.TemporaryFile

    def create(*args, **kwargs):
        if len(streams) == position:
            raise OSError("private temporary path")
        return original(*args, **kwargs)

    monkeypatch.setattr(module.tempfile, "TemporaryFile", create)
    with pytest.raises(
        module.WorkerExecutionError, match="^worker_setup_failed$"
    ) as caught:
        module.observe_worker(_command("raise SystemExit(0)"))
    assert processes == []
    assert all(stream.closed for stream in streams)
    assert json.loads(caught.value.progress)["pid"] is None


def test_missing_executable_does_not_expose_command(capsys):
    module = _module()
    with pytest.raises(module.WorkerExecutionError) as caught:
        module.observe_worker(("/private/invented-missing-worker",))
    assert caught.value.check_id == "worker_launch_failed"
    assert "private" not in str(caught.value) + caught.value.progress.decode()
    assert capsys.readouterr() == ("", "")
