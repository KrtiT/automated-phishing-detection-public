"""Actual externally reaped children never acquire zero-exit acceptance."""

import os
import signal

import pytest
from test_external_source_process_observation import context

from automated_phishing_detection import owned_worker

_CHILDREN = (
    "raise SystemExit(0)",
    "raise SystemExit(17)",
    "import os,signal;os.kill(os.getpid(),signal.SIGTERM)",
)


@pytest.mark.parametrize("code", _CHILDREN)
def test_external_reaper_means_unknown_even_when_completion_marker_exists(
    tmp_path, monkeypatch, code
):
    api, case = context(tmp_path, monkeypatch)
    case.command = (*case.command[:2], code)
    original = owned_worker.observe_owned_exit
    stolen = []

    def reap_first(process, *, block):
        assert not stolen
        stolen.append(os.waitpid(process.pid, 0))
        return original(process, block=block)

    def forbidden(*args, **kwargs):
        pytest.fail("lost process ownership reached completion or a new signal")

    monkeypatch.setattr(owned_worker, "observe_owned_exit", reap_first)
    monkeypatch.setattr(owned_worker.os, "kill", forbidden)
    monkeypatch.setattr(api, "verify_external_completion_snapshot", forbidden)
    with pytest.raises(Exception) as rejected:
        api._run_observed_external(case.binding, case.paths, case.handoff)
    assert len(stolen) == 1
    worker = rejected.value.external_failure.worker
    assert worker.exit.exit_observed is False
    assert worker.exit.exit_code is None


def test_observed_signal_exit_is_retained_without_saved_acceptance(
    tmp_path, monkeypatch
):
    api, case = context(tmp_path, monkeypatch)
    case.command = (*case.command[:2], _CHILDREN[-1])
    monkeypatch.setattr(
        api,
        "verify_external_completion_snapshot",
        lambda *args, **kwargs: pytest.fail("signal exit reached saved acceptance"),
    )
    with pytest.raises(Exception) as rejected:
        api._run_observed_external(case.binding, case.paths, case.handoff)
    worker = rejected.value.external_failure.worker
    assert worker.exit.exit_observed is True
    assert worker.exit.exit_code == -signal.SIGTERM
