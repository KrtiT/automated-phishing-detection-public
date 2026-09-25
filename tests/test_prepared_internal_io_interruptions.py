"""Actual SIGINT cannot escape prepared-only borrowed I/O ownership windows."""

import os
import signal

import pytest
from prepared_internal_fixtures import restored_case
from study_preparation_runner_fixtures import (
    inputs,
    preparation_api,
    preparation_case,
    runner,
)
from test_prepared_internal_runner import module
from test_study_preparation_io_interruptions import signal_after_open

from automated_phishing_detection import _internal_scientific_io, source_runner

__all__ = ["inputs", "preparation_api", "preparation_case", "runner"]


@pytest.fixture
def io_case(preparation_api, preparation_case, inputs):
    return restored_case(preparation_api, preparation_case, inputs)


def arm_call(monkeypatch, owner, name):
    original, active = getattr(owner, name), []

    def wrapped(*args, **kwargs):
        active.append(True)
        try:
            return original(*args, **kwargs)
        finally:
            active.pop()

    monkeypatch.setattr(owner, name, wrapped)
    return lambda: bool(active)


def failing_summary(*args):
    raise ValueError("private summary failure")


def signal_scope(monkeypatch, stage):
    if stage == "public":
        return "sources.json", None
    if stage == "final_binding":
        calls, active = [], []

        def recheck(binding):
            calls.append(binding)
            active.append(len(calls) == 2)
            try:
                source_runner._read_file_once(binding.root / "data/sources.json")
            finally:
                active.pop()

        monkeypatch.setattr(source_runner, "recheck_binding", recheck)
        return "sources.json", lambda: bool(active and active[-1])
    if stage in {"retain_failure_progress", "record_failure"}:
        monkeypatch.setattr(source_runner, "_secondary", failing_summary)
    owner = _internal_scientific_io if stage == "start" else source_runner
    return "/", arm_call(monkeypatch, owner, stage)


@pytest.mark.parametrize(
    "stage",
    [
        "public",
        "_prepared_output_paths",
        "retain_source_checkpoints",
        "start",
        "final_binding",
        "publish_completion",
        "retain_failure_progress",
        "record_failure",
    ],
)
def test_prepared_borrowed_io_defers_signal_through_owned_close(
    io_case, monkeypatch, stage
):
    case = io_case
    target, enabled = signal_scope(monkeypatch, stage)
    with signal_after_open(monkeypatch, target, enabled) as (opened, closed):
        with pytest.raises(KeyboardInterrupt):
            module()._run_bound_prepared_internal(
                case.binding, case.paths, case.preparation
            )
        assert len(opened) == 1 and opened[0] in closed
        with pytest.raises(OSError):
            os.fstat(opened[0])


def test_prepared_reservation_signal_keeps_assignment_for_failure_record(
    io_case, monkeypatch
):
    case, reserved = io_case, []
    original = source_runner.reserve_attempt

    def reserve(*args, **kwargs):
        attempt = original(*args, **kwargs)
        reserved.append(attempt)
        signal.raise_signal(signal.SIGINT)
        return attempt

    monkeypatch.setattr(source_runner, "reserve_attempt", reserve)
    with pytest.raises(KeyboardInterrupt):
        module()._run_bound_prepared_internal(
            case.binding, case.paths, case.preparation
        )
    assert len(reserved) == 1
    assert (case.paths.attempt / "outcome.json").is_file()
    assert (case.paths.attempt / "failure-progress.json").is_file()
    assert not case.session.primary.scorer.urls
