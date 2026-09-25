import os
import signal

import pytest
from test_owned_worker import _command
from test_owned_worker_failures import owned_resources as owned_resources


@pytest.fixture
def restoration_interrupt(monkeypatch):
    first, second, received = KeyboardInterrupt(), KeyboardInterrupt(), []
    install = signal.signal

    def handler(number, frame):
        received.append(number)
        raise first if len(received) == 1 else second

    def restore(number, callback):
        previous = install(number, callback)
        if callback is handler:
            os.kill(os.getpid(), signal.SIGINT)
        return previous

    previous = install(signal.SIGINT, handler)
    monkeypatch.setattr(signal, "signal", restore)
    try:
        yield first, handler, received
    finally:
        install(signal.SIGINT, previous)


def test_second_sigint_at_handler_restore_preserves_first(
    owned_resources, restoration_interrupt
):
    module, processes, streams = owned_resources
    first, handler, received = restoration_interrupt
    program = (
        "import os, signal, time; time.sleep(0.03); "
        "os.kill(os.getppid(), signal.SIGINT); time.sleep(60)"
    )
    with pytest.raises(KeyboardInterrupt) as caught:
        module.observe_worker(_command(program))
    assert caught.value is first
    assert received == [signal.SIGINT, signal.SIGINT]
    assert signal.getsignal(signal.SIGINT) is handler
    assert b'"failure":"parent_interrupted"' in caught.value.progress
    assert all(stream.closed for stream in streams)
    with pytest.raises(ChildProcessError):
        os.waitpid(processes[0].pid, os.WNOHANG)
