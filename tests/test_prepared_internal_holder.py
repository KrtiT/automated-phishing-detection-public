"""Holder registration protects ownership without deferring pure restoration."""

import signal
from contextlib import contextmanager

import pytest
from study_preparation_runner_fixtures import (
    inputs,
    preparation_api,
    preparation_case,
    runner,
)
from test_prepared_internal_process import _held_result, process_case
from test_prepared_internal_runner import module

from automated_phishing_detection import study_preparation_transport

__all__ = ["inputs", "preparation_api", "preparation_case", "runner", "process_case"]


def test_real_sigint_during_restore_is_immediate(process_case, monkeypatch):
    api, case, continued = module(), process_case, []
    original = study_preparation_transport._restore

    def restore(*args, **kwargs):
        signal.raise_signal(signal.SIGINT)
        continued.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(study_preparation_transport, "_restore", restore)
    with pytest.raises(KeyboardInterrupt):
        _held_result(api, case)
    assert continued == []
    assert not case.paths.attempt.exists()


def test_entry_failure_does_not_invoke_worker(process_case, monkeypatch):
    api, case = module(), process_case
    original = ValueError("entry failed")

    @contextmanager
    def broken(*args, **kwargs):
        raise original
        yield

    monkeypatch.setattr(api, "hold_study_preparation", broken)
    with pytest.raises(ValueError) as caught:
        _held_result(api, case)
    assert caught.value is original
    assert not case.paths.attempt.exists()
