import json
import os
import signal

import pytest
from study_admission_fixtures import child_command, frame, module
from test_study_admission_worker import observe

from automated_phishing_detection import owned_worker


@pytest.mark.parametrize("boundary", ("pipe", "set_blocking", "write"))
def test_setup_interrupt_registers_and_closes_both_pipe_ends(monkeypatch, boundary):
    allocated = []
    original_pipe = os.pipe
    original = getattr(os, boundary)

    def pipe():
        descriptors = original_pipe()
        allocated.extend(descriptors)
        if boundary == "pipe":
            signal.raise_signal(signal.SIGINT)
        return descriptors

    def interrupt(*args):
        value = original(*args)
        signal.raise_signal(signal.SIGINT)
        return value

    monkeypatch.setattr(os, "pipe", pipe)
    if boundary != "pipe":
        monkeypatch.setattr(os, boundary, interrupt)
    with pytest.raises(KeyboardInterrupt):
        with module().ParentAdmission(frame()):
            pytest.fail("interrupted channel must not launch")
    for descriptor in allocated:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_setup_mode_error_closes_unconfigured_write_end(monkeypatch):
    allocated = []
    original = os.pipe

    def pipe():
        allocated.extend(original())
        return tuple(allocated)

    def failure(*args):
        raise OSError("mode setup failed")

    monkeypatch.setattr(os, "pipe", pipe)
    monkeypatch.setattr(os, "set_blocking", failure)
    with pytest.raises(OSError):
        with module().ParentAdmission(frame()):
            pytest.fail("failed setup must not launch")
    try:
        for descriptor in allocated:
            with pytest.raises(OSError):
                os.fstat(descriptor)
    finally:
        for descriptor in allocated:
            try:
                os.close(descriptor)
            except OSError:
                pass


def test_signal_at_popen_return_retains_admitted_actual_process(monkeypatch):
    arguments = child_command()
    admission = module().ParentAdmission(frame(arguments))
    original = owned_worker.subprocess.Popen

    def interrupt(*args, **kwargs):
        process = original(*args, **kwargs)
        signal.raise_signal(signal.SIGINT)
        return process

    monkeypatch.setattr(owned_worker.subprocess, "Popen", interrupt)
    with pytest.raises(KeyboardInterrupt) as caught:
        observe(arguments, admission)
    progress = json.loads(caught.value.progress)
    assert progress["pid"] == admission.pid
    assert progress["exit_observed"] is admission.exit_observed is True
    assert admission.pipe.descriptors == set()


def test_reaped_child_is_unknown_in_admission_observation(monkeypatch):
    arguments = child_command()
    admission = module().ParentAdmission(frame(arguments))
    original = owned_worker.subprocess.Popen

    def reap(*args, **kwargs):
        process = original(*args, **kwargs)
        os.waitpid(process.pid, 0)
        process.returncode = 0
        return process

    monkeypatch.setattr(owned_worker.subprocess, "Popen", reap)
    observed = observe(arguments, admission)
    assert observed.exit.exit_observed is admission.exit_observed is False
    assert observed.exit.exit_code is admission.exit_code is None
    assert admission.recorded is True
    assert admission.pipe.descriptors == set()
