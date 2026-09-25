import json
import os
import signal

import pytest
from test_owned_worker import _command
from test_owned_worker_failures import owned_resources as owned_resources


def _assert_joined_interrupt(error, processes, streams):
    progress = json.loads(error.progress)
    assert progress["pid"] == processes[0].pid
    assert progress["exit_observed"] is True
    assert progress["exit_code"] == -signal.SIGKILL
    assert progress["failure"] == "parent_interrupted"
    assert all(stream.closed for stream in streams)


@pytest.mark.parametrize("when", ["launch", "wait"])
def test_interrupt_joins_child_and_preserves_original(
    monkeypatch, owned_resources, when
):
    module, processes, streams = owned_resources
    failure = KeyboardInterrupt("private interrupt diagnostic")
    original = module.observe_owned_exit

    def interrupt(process, *, block=False):
        monkeypatch.setattr(module, "observe_owned_exit", original)
        raise failure

    if when == "launch":
        original_launch = module.subprocess.Popen

        def launch(*args, **kwargs):
            process = original_launch(*args, **kwargs)
            os.kill(os.getpid(), signal.SIGINT)
            return process

        monkeypatch.setattr(module.subprocess, "Popen", launch)
    else:
        monkeypatch.setattr(module, "observe_owned_exit", interrupt)
    with pytest.raises(KeyboardInterrupt) as caught:
        module.observe_worker(_command("import time; time.sleep(60)"))
    if when == "wait":
        assert caught.value is failure
    _assert_joined_interrupt(caught.value, processes, streams)


def test_interrupt_after_kernel_reap_never_signals_unknown(
    monkeypatch, owned_resources
):
    module, processes, streams = owned_resources
    failure = KeyboardInterrupt()
    original = module.observe_owned_exit
    observations = []

    def interrupt(process, *, block=False):
        observations.append(block)
        if len(observations) == 1:
            os.waitpid(process.pid, 0)
            raise failure
        return original(process, block=block)

    def forbidden(*args, **kwargs):
        pytest.fail("a consumed kernel status must never permit signaling")

    monkeypatch.setattr(module, "observe_owned_exit", interrupt)
    monkeypatch.setattr(module.os, "kill", forbidden)
    with pytest.raises(KeyboardInterrupt) as caught:
        module.observe_worker(_command("raise SystemExit(17)"))
    assert caught.value is failure
    progress = json.loads(caught.value.progress)
    assert observations == [True, False]
    assert progress["exit_observed"] is False
    assert progress["exit_code"] is None
    assert progress["signals"] == []
    assert all(stream.closed for stream in streams)


def test_interrupt_before_launch_has_no_child(monkeypatch, owned_resources):
    module, processes, streams = owned_resources
    failure = KeyboardInterrupt()

    def interrupt(*args, **kwargs):
        raise failure

    monkeypatch.setattr(module.subprocess, "Popen", interrupt)
    with pytest.raises(KeyboardInterrupt) as caught:
        module.observe_worker(_command("raise SystemExit(0)"))
    assert caught.value is failure
    assert processes == []
    assert json.loads(caught.value.progress)["pid"] is None
    assert all(stream.closed for stream in streams)


@pytest.mark.parametrize("cleanup_failure", [OSError("private"), KeyboardInterrupt()])
def test_cleanup_failure_preserves_first_interrupt(
    monkeypatch, owned_resources, cleanup_failure
):
    module, processes, streams = owned_resources
    failure = KeyboardInterrupt()
    original = module.observe_owned_exit

    def interrupt(process, *, block=False):
        monkeypatch.setattr(module, "observe_owned_exit", original)
        raise failure

    def fail(*args, **kwargs):
        raise cleanup_failure

    monkeypatch.setattr(module, "observe_owned_exit", interrupt)
    monkeypatch.setattr(module.os, "kill", fail)
    with pytest.raises(KeyboardInterrupt) as caught:
        module.observe_worker(_command("import time; time.sleep(60)"))
    assert caught.value is failure
    progress = json.loads(caught.value.progress)
    assert progress["failure"] == "parent_interrupted"
    assert progress["cleanup_failures"] == ["worker_cleanup_failed"]
    assert all(stream.closed for stream in streams)


@pytest.mark.parametrize("position", [1, 2])
def test_interrupt_after_stream_allocation_closes_owned_descriptors(
    monkeypatch, owned_resources, position
):
    module, processes, streams = owned_resources
    original = module.tempfile.TemporaryFile

    def create(*args, **kwargs):
        stream = original(*args, **kwargs)
        if len(streams) == position:
            os.kill(os.getpid(), signal.SIGINT)
        return stream

    monkeypatch.setattr(module.tempfile, "TemporaryFile", create)
    with pytest.raises(KeyboardInterrupt) as caught:
        module.observe_worker(_command("raise SystemExit(0)"))
    assert processes == []
    assert all(stream.closed for stream in streams)
    assert json.loads(caught.value.progress)["failure"] == "parent_interrupted"


def test_interrupt_that_rejects_progress_attachment_retains_identity(
    monkeypatch, owned_resources
):
    module, processes, streams = owned_resources

    class UnmodifiableInterrupt(KeyboardInterrupt):
        def __setattr__(self, name, value):
            if name == "progress":
                raise RuntimeError("private attachment failure")
            super().__setattr__(name, value)

    failure, original = UnmodifiableInterrupt(), module.observe_owned_exit

    def interrupt(process, *, block=False):
        monkeypatch.setattr(module, "observe_owned_exit", original)
        raise failure

    monkeypatch.setattr(module, "observe_owned_exit", interrupt)
    with pytest.raises(UnmodifiableInterrupt) as caught:
        module.observe_worker(_command("import time; time.sleep(60)"))
    assert caught.value is failure
    assert all(stream.closed for stream in streams)
    with pytest.raises(ChildProcessError):
        os.waitpid(processes[0].pid, os.WNOHANG)
