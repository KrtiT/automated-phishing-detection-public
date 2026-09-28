import inspect
import json
import os
import signal

import pytest
from study_admission_fixtures import child_command, command, frame, module

from automated_phishing_detection import owned_worker


def observe(arguments, admission):
    assert hasattr(owned_worker, "_observe_study_worker"), (
        "missing private study worker"
    )
    return owned_worker._observe_study_worker(arguments, admission)


def test_private_worker_retains_actual_launch_exit_and_only_inherits_read_end(
    monkeypatch,
):
    arguments, launched = child_command(), []
    admission = module().ParentAdmission(frame(arguments))
    original = owned_worker.subprocess.Popen

    def launch(*args, **kwargs):
        assert kwargs["pass_fds"] == (admission.read_fd,)
        assert kwargs["env"]["APD_STUDY_ADMISSION_FD"] == str(admission.read_fd)
        assert "APD_STALE" not in kwargs["env"]
        process = original(*args, **kwargs)
        launched.append(process.pid)
        return process

    monkeypatch.setenv("APD_STALE", "discard")
    monkeypatch.setattr(owned_worker.subprocess, "Popen", launch)
    result = observe(arguments, admission)
    assert result.exit.pid == admission.pid == launched[0]
    assert result.exit.exit_observed is admission.exit_observed is True
    assert result.exit.exit_code == admission.exit_code == 0
    assert admission.pipe.descriptors == set()


def test_failed_launch_never_invents_actual_pid_or_exit(tmp_path):
    arguments = (str(tmp_path / "missing-worker"),)
    admission = module().ParentAdmission(frame(arguments))
    with pytest.raises(owned_worker.WorkerExecutionError) as caught:
        observe(arguments, admission)
    progress = json.loads(caught.value.progress)
    assert admission.pid is progress["pid"] is None
    assert admission.exit_observed is progress["exit_observed"] is False
    assert admission.exit_code is progress["exit_code"] is None
    assert admission.pipe.descriptors == set()


@pytest.mark.parametrize("when", ("launched", "observed"))
def test_callback_interrupt_preserves_first_error_and_real_child_evidence(when):
    arguments = command(
        "import signal; signal.pause()" if when == "launched" else "pass"
    )
    failure = KeyboardInterrupt("first")

    def interrupt(*unused):
        raise failure

    admission = module().ParentAdmission(frame(arguments), **{f"on_{when}": interrupt})
    with pytest.raises(KeyboardInterrupt) as caught:
        observe(arguments, admission)
    assert caught.value is failure
    progress = json.loads(caught.value.progress)
    assert progress["pid"] == admission.pid
    assert progress["exit_observed"] is admission.exit_observed is True
    assert progress["exit_code"] == (-signal.SIGKILL if when == "launched" else 0)
    assert admission.pipe.descriptors == set()
    with pytest.raises(ChildProcessError):
        os.waitpid(admission.pid, os.WNOHANG)


def test_worker_rejects_mismatched_command_before_launch(monkeypatch):
    admission = module().ParentAdmission(frame())

    def forbidden(*unused, **keywords):
        pytest.fail("mismatched admission cannot launch")

    monkeypatch.setattr(owned_worker.subprocess, "Popen", forbidden)
    with pytest.raises(owned_worker.WorkerExecutionError):
        observe(command("different"), admission)
    assert admission.pid is None


def test_original_worker_api_remains_closed():
    assert list(inspect.signature(owned_worker.observe_worker).parameters) == [
        "command"
    ]
