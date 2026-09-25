import os
from contextlib import contextmanager

import pytest
from test_owned_worker import _command
from test_owned_worker_failures import owned_resources as owned_resources


def test_unknown_keeps_returncode_and_disarms_destructor(monkeypatch, owned_resources):
    module, processes, streams = owned_resources
    original = module.subprocess.Popen

    def launch(*args, **kwargs):
        process = original(*args, **kwargs)
        os.waitpid(process.pid, 0)
        return process

    def forbidden(*args, **kwargs):
        pytest.fail("a terminal unknown child cannot be waited again")

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    observed = module.observe_worker(_command("raise SystemExit(17)"))
    assert observed.exit.exit_observed is False
    assert observed.exit.exit_code is None
    assert processes[0].returncode is None
    monkeypatch.setattr(processes[0], "_internal_poll", forbidden)
    processes[0].__del__()


def test_descriptor_cleanup_failure_retains_observed_exit(monkeypatch, owned_resources):
    module, processes, streams = owned_resources
    original = module.tempfile.TemporaryFile

    @contextmanager
    def create():
        stream = original()
        try:
            yield stream
        finally:
            stream.close()
            raise OSError("private descriptor diagnostic")

    monkeypatch.setattr(module.tempfile, "TemporaryFile", create)
    with pytest.raises(
        module.WorkerExecutionError, match="^worker_cleanup_failed$"
    ) as caught:
        module.observe_worker(_command("raise SystemExit(0)"))
    assert b'"exit_observed":true' in caught.value.progress
    assert b'"exit_code":0' in caught.value.progress
    assert all(stream.closed for stream in streams)
