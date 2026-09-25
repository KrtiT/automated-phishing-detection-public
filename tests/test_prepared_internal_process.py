"""Only an owned successful observation reaches retained-preparation acceptance."""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from prepared_internal_fixtures import restored_case
from study_preparation_runner_fixtures import (
    inputs,
    preparation_api,
    preparation_case,
    runner,
)
from test_prepared_internal_runner import module

from automated_phishing_detection._owned_process_exit import OwnedProcessExit
from automated_phishing_detection._process_support import command_hash
from automated_phishing_detection.owned_worker import WorkerObservation

__all__ = ["inputs", "preparation_api", "preparation_case", "runner"]


@pytest.fixture
def process_case(preparation_api, preparation_case, inputs):
    return restored_case(preparation_api, preparation_case, inputs)


def observe_api():
    api = module()
    assert hasattr(api, "_run_observed_prepared_internal"), "missing prepared parent"
    return api


@pytest.mark.parametrize(
    "exit_value",
    [
        OwnedProcessExit(11, False, 0),
        OwnedProcessExit(11, True, 17),
        OwnedProcessExit(11, True, -9),
        OwnedProcessExit(11, True, False),
    ],
)
def test_unaccepted_prepared_worker_never_reaches_saved_verifier(
    process_case, monkeypatch, exit_value
):
    api, case = observe_api(), process_case
    monkeypatch.setattr(
        api,
        "observe_worker",
        lambda command: WorkerObservation(
            command_hash(command), exit_value, "a" * 64, "b" * 64
        ),
    )
    monkeypatch.setattr(
        api,
        "verify_prepared_internal_completion_snapshot",
        lambda *args, **kwargs: pytest.fail("unaccepted exit reached verification"),
    )
    with pytest.raises(api.SourceExecutionError, match="worker_exit") as caught:
        api._run_observed_prepared_internal(
            case.binding, case.paths, preparation=case.preparation
        )
    assert caught.value.worker_failure.worker.exit is exit_value


def test_successful_observation_passes_same_parent_preparation(
    process_case, monkeypatch
):
    api, case, seen = observe_api(), process_case, []
    observation = None

    def observe(command):
        nonlocal observation
        observation = WorkerObservation(
            command_hash(command), OwnedProcessExit(11, True, 0), "a" * 64, "b" * 64
        )
        return observation

    def verify(binding, paths, **kwargs):
        assert binding is case.binding and paths is case.paths
        assert kwargs == {"preparation": case.preparation, "producer_exit_code": 0}
        seen.append(True)
        return SimpleNamespace(public_summary={"accepted": True})

    monkeypatch.setattr(api, "observe_worker", observe)
    monkeypatch.setattr(api, "verify_prepared_internal_completion_snapshot", verify)
    result = api._run_observed_prepared_internal(
        case.binding, case.paths, preparation=case.preparation
    )
    assert result.worker is observation and seen == [True]


@pytest.mark.parametrize(
    "kind", [KeyboardInterrupt, SystemExit, asyncio.CancelledError]
)
def test_observer_interruption_does_not_start_another_worker(
    process_case, monkeypatch, kind
):
    api, case, calls = observe_api(), process_case, []
    original = kind()

    def observe(command):
        calls.append(command)
        raise original

    monkeypatch.setattr(api, "observe_worker", observe)
    with pytest.raises(kind) as caught:
        api._run_observed_prepared_internal(
            case.binding, case.paths, preparation=case.preparation
        )
    assert caught.value is original and len(calls) == 1


def test_preparation_holder_lives_through_parent_verification(
    process_case, monkeypatch
):
    api, case, events = observe_api(), process_case, []
    assert hasattr(api, "_run_held"), "missing owned preparation holder"

    @contextmanager
    def hold(directory, **kwargs):
        assert directory == case.paths.preparation
        assert kwargs["expected_identity"] == case.identity
        events.append("entered")
        yield case.preparation
        events.append("closed")

    def observed(binding, paths, **kwargs):
        assert events == ["entered"]
        assert kwargs["preparation"] is case.preparation
        events.append("verified")
        return "accepted"

    monkeypatch.setattr(api, "hold_study_preparation", hold)
    monkeypatch.setattr(api, "_run_observed_prepared_internal", observed)
    assert _held_result(api, case) == "accepted"
    assert events == ["entered", "verified", "closed"]


def _held_result(api, case):
    context = (case.identity, case.source, case.buffers, case.profile)
    return api._run_held(
        case.binding,
        case.paths,
        context,
        case.preparation.reservation_sha256,
        case.preparation.completion_sha256,
        True,
    )
