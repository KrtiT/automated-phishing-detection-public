"""Actual signals must not replace earlier progress during parent unwinding."""

import os
import signal
from contextlib import contextmanager

import pytest
from operational_cell_runner_fixtures import execute, orchestration, setup
from operational_input_fixtures import candidates, manifests

__all__ = ["candidates", "manifests"]


def test_writer_cleanup_signal_keeps_original_supervisor_progress(
    tmp_path, manifests, monkeypatch
):
    case = setup(tmp_path, manifests, monkeypatch)
    orchestration(case, monkeypatch)
    earlier = ValueError("actual supervisor failure")
    earlier.progress = b"actual partial bytes"
    original = case.module.held_attempt_writer

    @contextmanager
    def late(*arguments, **keywords):
        try:
            with original(*arguments, **keywords) as writer:
                yield writer
        finally:
            os.kill(os.getpid(), signal.SIGINT)

    async def fail(*arguments, **keywords):
        raise earlier

    monkeypatch.setattr(case.module, "held_attempt_writer", late)
    monkeypatch.setattr(case.module, "_observe_pair_with_writer", fail)
    with pytest.raises(KeyboardInterrupt) as caught:
        execute(case)
    assert caught.value.progress == earlier.progress
    assert caught.value.operational_failure.observation is None


def test_writer_final_signal_retains_actual_returned_observation(
    tmp_path, manifests, monkeypatch
):
    case = setup(tmp_path, manifests, monkeypatch)
    orchestration(case, monkeypatch)
    original = case.module.held_attempt_writer

    @contextmanager
    def late(*arguments, **keywords):
        with original(*arguments, **keywords) as writer:
            yield writer
        os.kill(os.getpid(), signal.SIGINT)

    monkeypatch.setattr(case.module, "held_attempt_writer", late)
    with pytest.raises(KeyboardInterrupt) as caught:
        execute(case)
    assert caught.value.operational_failure.observation is case.observation
    assert caught.value.operational_failure.candidate is None
    assert "complete" not in case.events


def test_new_interruption_during_failure_persistence_is_not_swallowed(
    tmp_path, manifests, monkeypatch
):
    from automated_phishing_detection import execution_receipt as receipt

    case = setup(tmp_path, manifests, monkeypatch)
    orchestration(case, monkeypatch)
    original = ValueError("ordinary")
    original.progress = b"actual bytes"

    async def fail(*arguments, **keywords):
        raise original

    def interrupted(*arguments, **keywords):
        os.kill(os.getpid(), signal.SIGINT)

    monkeypatch.setattr(case.module, "_observe_pair_with_writer", fail)
    monkeypatch.setattr(receipt, "record_failure", interrupted)
    with pytest.raises(KeyboardInterrupt) as caught:
        execute(case)
    assert caught.value.progress == original.progress
    assert caught.value.operational_failure.attempt is case.attempt
