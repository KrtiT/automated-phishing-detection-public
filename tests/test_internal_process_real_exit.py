"""Actual child outcomes, not installed markers or cached zeros, gate acceptance."""

import os
import sys
from types import SimpleNamespace

import pytest
from test_internal_process_observation import _call, _ready
from test_source_runner import inputs as inputs
from test_source_runner import runner as runner

from automated_phishing_detection import owned_worker, source_completion


def _command(marker, termination):
    program = (
        "import os, signal, sys; from pathlib import Path; "
        "Path(sys.argv[1]).write_text('installed marker'); "
        "print('private-worker-canary', file=sys.stderr); "
    )
    program += (
        "os.kill(os.getpid(), signal.SIGTERM)"
        if termination == "signal"
        else f"raise SystemExit({termination})"
    )
    return sys.executable, "-c", program, str(marker)


def _external_reaper(monkeypatch):
    original = owned_worker.subprocess.Popen
    exits = []

    def launch(*args, **kwargs):
        process = original(*args, **kwargs)
        pid, status = os.waitpid(process.pid, 0)
        assert pid == process.pid
        exits.append(os.waitstatus_to_exitcode(status))
        process.returncode = 0
        return process

    monkeypatch.setattr(owned_worker.subprocess, "Popen", launch)
    return exits


def _verify(monkeypatch, accepted):
    calls = []

    def verify(*args, **kwargs):
        assert accepted, "unobserved or failed worker reached verification"
        assert kwargs == {"producer_exit_code": 0}
        calls.append(True)
        return SimpleNamespace(public_summary={"verified": True})

    monkeypatch.setattr(
        source_completion, "verify_internal_completion_snapshot", verify
    )
    return calls


@pytest.mark.parametrize("termination", [0, 17, "signal"])
@pytest.mark.parametrize("reaped", [False, True])
def test_marker_cannot_replace_actual_owned_success(
    runner, inputs, monkeypatch, capsys, termination, reaped
):
    binding, paths, unused_session, unused_events = inputs
    _ready(runner, binding, monkeypatch)
    command = _command(paths.public_summary, termination)
    monkeypatch.setattr(runner, "_worker_command", lambda *args: command)
    actual_exits = _external_reaper(monkeypatch) if reaped else []
    accepted = termination == 0 and not reaped
    reads = _verify(monkeypatch, accepted)
    if accepted:
        result = _call(runner.run_internal_process_with_evidence, binding, paths)
        assert result.worker.exit.exit_observed is True
        assert result.worker.exit.exit_code == 0
    else:
        with pytest.raises(runner.SourceExecutionError, match="worker_exit") as caught:
            _call(runner.run_internal_process_with_evidence, binding, paths)
        assert "private-worker-canary" not in str(caught.value)
    assert reads == ([True] if accepted else [])
    assert paths.public_summary.read_text() == "installed marker"
    assert actual_exits == (
        [-15 if termination == "signal" else termination] if reaped else []
    )
    assert capsys.readouterr().err == ""
