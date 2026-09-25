"""Failure state retains only actual objects obtained by the private composition."""

import json
import os
import signal
from contextlib import contextmanager

import pytest
from operational_cell_runner_fixtures import execute, holder, orchestration, setup
from operational_input_fixtures import candidates, manifests

from automated_phishing_detection import execution_receipt as receipt

__all__ = ["candidates", "manifests"]


@pytest.mark.parametrize(
    "error", [ValueError("private"), KeyboardInterrupt("first"), SystemExit(19)]
)
def test_observation_failure_consumes_attempt_without_fabricating_exit(
    tmp_path, manifests, monkeypatch, error
):
    case = setup(tmp_path, manifests, monkeypatch)
    orchestration(case, monkeypatch)
    error.progress = b"actual partial supervisor bytes"

    async def fail(*arguments, **keywords):
        raise error

    monkeypatch.setattr(case.module, "_observe_pair_with_writer", fail)
    with pytest.raises(BaseException) as caught:
        execute(case)
    if not isinstance(error, Exception):
        assert caught.value is error
    failed = caught.value.operational_failure
    assert failed.attempt is case.attempt
    assert failed.observation is failed.working is failed.candidate is None
    assert caught.value.progress == error.progress
    assert (
        json.loads((case.paths.attempt / "outcome.json").read_bytes())["status"]
        == "failed"
    )
    assert not case.paths.public_summary.exists()


def test_late_holder_interruption_retains_observation_and_candidate(
    tmp_path, manifests, monkeypatch
):
    case = setup(tmp_path, manifests, monkeypatch)
    orchestration(case, monkeypatch)
    original = holder(case)

    @contextmanager
    def late(*arguments, **keywords):
        with original(*arguments, **keywords) as held:
            yield held
        os.kill(os.getpid(), signal.SIGINT)

    monkeypatch.setattr(case.module, "hold_operational_cell", late)
    with pytest.raises(KeyboardInterrupt) as caught:
        execute(case)
    failed = caught.value.operational_failure
    assert failed.observation is case.observation
    assert failed.candidate is case.snapshot and failed.publishing
    assert not (case.paths.attempt / "outcome.json").exists()


def test_original_interruption_survives_failure_persistence(
    tmp_path, manifests, monkeypatch
):
    case = setup(tmp_path, manifests, monkeypatch)
    orchestration(case, monkeypatch)
    first = KeyboardInterrupt("first")

    async def fail(*arguments, **keywords):
        raise first

    def second(*arguments, **keywords):
        os.kill(os.getpid(), signal.SIGINT)

    monkeypatch.setattr(case.module, "_observe_pair_with_writer", fail)
    monkeypatch.setattr(receipt, "record_failure", second)
    with pytest.raises(KeyboardInterrupt) as caught:
        execute(case)
    assert (
        caught.value is first
        and caught.value.operational_failure.attempt is case.attempt
    )


def test_reservation_return_and_assignment_precede_deferred_signal(
    tmp_path, manifests, monkeypatch
):
    case = setup(tmp_path, manifests, monkeypatch)
    original = receipt.reserve_attempt
    obtained = []

    def reserve(*arguments, **keywords):
        attempt = original(*arguments, **keywords)
        obtained.append(attempt)
        os.kill(os.getpid(), signal.SIGINT)
        return attempt

    monkeypatch.setattr(receipt, "reserve_attempt", reserve)
    with pytest.raises(KeyboardInterrupt) as caught:
        execute(case)
    assert caught.value.operational_failure.attempt is obtained[0]
    assert caught.value.operational_failure.observation is None
    assert (
        json.loads((case.paths.attempt / "outcome.json").read_bytes())["status"]
        == "failed"
    )


def test_actual_cleanup_signal_keeps_existing_operational_failure(
    tmp_path, manifests, monkeypatch
):
    case = setup(tmp_path, manifests, monkeypatch)
    orchestration(case, monkeypatch)
    original = ValueError("earlier")
    original.operational_failure = object()
    original.progress = b"actual progress"

    async def fail(*arguments, **keywords):
        raise original

    @contextmanager
    def late(*arguments, **keywords):
        try:
            yield case.completer
        finally:
            os.kill(os.getpid(), signal.SIGINT)

    monkeypatch.setattr(case.module, "_observe_pair_with_writer", fail)
    monkeypatch.setattr(case.module, "hold_operational_cell", late)
    with pytest.raises(KeyboardInterrupt) as caught:
        execute(case)
    assert caught.value.operational_failure is original.operational_failure
    assert caught.value.progress == original.progress
