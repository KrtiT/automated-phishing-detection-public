import inspect
import os
import signal
import sys

import pytest
from test_owned_worker import _command
from test_owned_worker_failures import owned_resources as owned_resources


def _interrupt_at(function, text):
    lines, start = inspect.getsourcelines(function)
    target = start + next(index for index, line in enumerate(lines) if text in line)

    def trace(frame, event, argument):
        if (
            event == "line"
            and frame.f_code is function.__code__
            and frame.f_lineno == target
        ):
            sys.settrace(None)
            os.kill(os.getpid(), signal.SIGINT)
        return trace

    return trace


@pytest.mark.parametrize("point", ["transition", "loop", "action"])
def test_second_sigint_at_cleanup_transition_preserves_first_and_reaps(
    monkeypatch, owned_resources, point
):
    module, processes, streams = owned_resources
    failure, original = KeyboardInterrupt(), module.observe_owned_exit

    def interrupt(process, *, block=False):
        monkeypatch.setattr(module, "observe_owned_exit", original)
        raise failure

    monkeypatch.setattr(module, "observe_owned_exit", interrupt)
    function, text = {
        "transition": (module.observe_worker, "worker.finish("),
        "loop": (module._Worker.finish, "for action in"),
        "action": (module._Worker.finish, "action()"),
    }[point]
    sys.settrace(_interrupt_at(function, text))
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            module.observe_worker(_command("import time; time.sleep(60)"))
    finally:
        sys.settrace(None)
    assert caught.value is failure
    assert all(stream.closed for stream in streams)
    with pytest.raises(ChildProcessError):
        os.waitpid(processes[0].pid, os.WNOHANG)


def test_real_signal_interrupts_unbounded_normal_wait(owned_resources):
    module, processes, streams = owned_resources
    program = (
        "import os, signal, time; time.sleep(0.03); "
        "os.kill(os.getppid(), signal.SIGINT); time.sleep(60)"
    )
    with pytest.raises(KeyboardInterrupt) as caught:
        module.observe_worker(_command(program))
    assert b'"failure":"parent_interrupted"' in caught.value.progress
    assert all(stream.closed for stream in streams)
    with pytest.raises(ChildProcessError):
        os.waitpid(processes[0].pid, os.WNOHANG)


@pytest.mark.parametrize("point", ["selection", "restoration"])
def test_first_signal_during_guard_exit_is_reported(owned_resources, point):
    module, processes, streams = owned_resources
    original = signal.getsignal(signal.SIGINT)
    text = "if selected is not error" if point == "selection" else "if self.enabled"
    sys.settrace(_interrupt_at(module._InterruptGuard.__exit__, text))
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            module.observe_worker(_command("raise SystemExit(0)"))
    finally:
        sys.settrace(None)
        restored = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, original)
    assert restored is original
    assert b'"failure":"parent_interrupted"' in caught.value.progress
    assert all(stream.closed for stream in streams)
    with pytest.raises(ChildProcessError):
        os.waitpid(processes[0].pid, os.WNOHANG)


@pytest.mark.parametrize("point", ["cleanup", "return"])
def test_first_signal_during_successful_cleanup_is_reported(owned_resources, point):
    module, processes, streams = owned_resources
    text = "worker.finish(" if point == "cleanup" else "return observed"
    sys.settrace(_interrupt_at(module.observe_worker, text))
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            module.observe_worker(_command("raise SystemExit(0)"))
    finally:
        sys.settrace(None)
    assert b'"failure":"parent_interrupted"' in caught.value.progress
    assert b'"exit_code":0' in caught.value.progress
    assert all(stream.closed for stream in streams)
    with pytest.raises(ChildProcessError):
        os.waitpid(processes[0].pid, os.WNOHANG)


def test_buffered_first_interrupt_survives_cleanup_exception(
    monkeypatch, owned_resources
):
    module, processes, streams = owned_resources
    first, second = KeyboardInterrupt(), KeyboardInterrupt()

    def handler(number, frame):
        raise first

    def stop(worker):
        os.kill(os.getpid(), signal.SIGINT)
        raise second

    monkeypatch.setattr(module._Worker, "stop", stop)
    original = signal.signal(signal.SIGINT, handler)
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            module.observe_worker(_command("raise SystemExit(0)"))
    finally:
        signal.signal(signal.SIGINT, original)
    assert caught.value is first
    assert all(stream.closed for stream in streams)


def test_signal_before_guard_run_never_launches_worker(owned_resources):
    module, processes, streams = owned_resources
    sys.settrace(_interrupt_at(module.observe_worker, "observed = interrupts.run"))
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            module.observe_worker(_command("raise SystemExit(0)"))
    finally:
        sys.settrace(None)
    assert processes == []
    assert streams == []
    assert b'"pid":null' in caught.value.progress
    assert b'"failure":"parent_interrupted"' in caught.value.progress
