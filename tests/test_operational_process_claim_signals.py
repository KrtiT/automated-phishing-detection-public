"""Initial claim association cannot replace the first interruption."""

import os
import signal

import pytest
from test_operational_process import inputs, operational
from test_operational_process_writer import observe

__all__ = ["operational"]


@pytest.mark.parametrize("phase", ["fail", "snapshot"])
@pytest.mark.parametrize("first", [KeyboardInterrupt("first"), SystemExit(23)])
def test_claim_association_preserves_first_interruption(
    operational, tmp_path, monkeypatch, phase, first
):
    attempt, options = inputs(tmp_path)

    def writer(*arguments):
        raise first

    def interrupted(*arguments):
        os.kill(os.getpid(), signal.SIGINT)

    monkeypatch.setattr(operational.Observations, phase, interrupted)
    with pytest.raises(BaseException) as caught:
        observe(operational, attempt, options, writer)
    assert caught.value is first
    assert operational.process_progress(caught.value) is None
    assert os.listdir(attempt.directory) == ["reservation.json"]


def test_two_actual_claim_signals_preserve_first_object(
    operational, tmp_path, monkeypatch
):
    attempt, options = inputs(tmp_path)
    captured = []

    def writer(*arguments):
        try:
            os.kill(os.getpid(), signal.SIGINT)
        except KeyboardInterrupt as error:
            captured.append(error)
            raise

    def interrupted(*arguments):
        os.kill(os.getpid(), signal.SIGINT)

    monkeypatch.setattr(operational.Observations, "snapshot", interrupted)
    with pytest.raises(KeyboardInterrupt) as caught:
        observe(operational, attempt, options, writer)
    assert caught.value is captured[0]
    assert operational.process_progress(caught.value) is None


def test_claim_association_does_not_swallow_interrupt_after_ordinary_error(
    operational, tmp_path, monkeypatch
):
    attempt, options = inputs(tmp_path)

    def writer(*arguments):
        raise ValueError("ordinary write failure")

    def interrupted(*arguments):
        os.kill(os.getpid(), signal.SIGINT)

    monkeypatch.setattr(operational.Observations, "fail", interrupted)
    with pytest.raises(KeyboardInterrupt):
        observe(operational, attempt, options, writer)
