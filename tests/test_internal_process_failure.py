"""Later rejection cannot discard the parent's already observed worker exit."""

import asyncio
import os
import signal
import sys
from dataclasses import FrozenInstanceError

import pytest
from test_internal_process_observation import _call, _observer, _ready
from test_source_runner import inputs as inputs
from test_source_runner import runner as runner

from automated_phishing_detection import source_completion
from automated_phishing_detection._owned_process_exit import OwnedProcessExit


def _setup(runner, inputs, monkeypatch, exit_result):
    binding, paths, unused_session, unused_events = inputs
    _ready(runner, binding, monkeypatch)
    observed = _observer(runner, monkeypatch, exit_result)
    return binding, paths, observed


def _assert_retained(error, observed, binding, stage):
    retained = error.worker_failure
    assert retained.worker is observed[0][1]
    assert retained.binding is binding
    assert retained.stage == stage
    with pytest.raises(FrozenInstanceError):
        retained.stage = "accepted"


@pytest.mark.parametrize(
    "kind", [ValueError, asyncio.CancelledError, KeyboardInterrupt, SystemExit]
)
def test_completion_failure_retains_observation_and_original_identity(
    runner, inputs, monkeypatch, kind
):
    binding, paths, observed = _setup(
        runner, inputs, monkeypatch, OwnedProcessExit(123, True, 0)
    )
    original = kind("private-exception-canary")

    def fail(*args, **kwargs):
        raise original

    monkeypatch.setattr(source_completion, "verify_internal_completion_snapshot", fail)
    with pytest.raises(kind) as caught:
        _call(runner.run_internal_process_with_evidence, binding, paths)
    assert caught.value is original
    _assert_retained(original, observed, binding, "completion_verification")


def _interrupt_after_observation(function):
    def trace(frame, event, argument):
        if (
            event == "line"
            and frame.f_code is function.__code__
            and frame.f_locals.get("observed") is not None
        ):
            sys.settrace(None)
            os.kill(os.getpid(), signal.SIGINT)
        return trace

    return trace


def test_first_interrupt_after_worker_return_keeps_actual_exit(
    runner, inputs, monkeypatch
):
    binding, paths, unused_session, unused_events = inputs
    _ready(runner, binding, monkeypatch)
    command = (sys.executable, "-c", "raise SystemExit(0)")
    monkeypatch.setattr(runner, "_worker_command", lambda *args: command)
    sys.settrace(_interrupt_after_observation(runner._run_observed_internal))
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            _call(runner.run_internal_process_with_evidence, binding, paths)
    finally:
        sys.settrace(None)
    retained = caught.value.worker_failure
    assert retained.stage == "worker_acceptance"
    assert retained.binding is binding
    assert retained.worker.exit.pid > 0
    assert retained.worker.exit.exit_observed is True
    assert retained.worker.exit.exit_code == 0


@pytest.mark.parametrize(
    "exit_result",
    [OwnedProcessExit(123, False, None), OwnedProcessExit(123, True, 17)],
)
def test_rejected_worker_retains_observation_without_completion_reads(
    runner, inputs, monkeypatch, exit_result
):
    binding, paths, observed = _setup(runner, inputs, monkeypatch, exit_result)

    def forbidden(*args, **kwargs):
        pytest.fail("unaccepted worker reached completion verification")

    monkeypatch.setattr(
        source_completion, "verify_internal_completion_snapshot", forbidden
    )
    with pytest.raises(runner.SourceExecutionError) as caught:
        _call(runner.run_internal_process_with_evidence, binding, paths)
    _assert_retained(caught.value, observed, binding, "worker_acceptance")
    assert caught.value.worker_failure.worker.exit is exit_result


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_refused_attachment_cannot_replace_original_interruption(
    runner, inputs, monkeypatch, failure
):
    class RefusesAttachment(KeyboardInterrupt):
        def __setattr__(self, name, value):
            raise failure("attachment refused")

    binding, paths, unused_observed = _setup(
        runner, inputs, monkeypatch, OwnedProcessExit(123, True, 0)
    )
    original = RefusesAttachment()

    def fail(*args, **kwargs):
        raise original

    monkeypatch.setattr(source_completion, "verify_internal_completion_snapshot", fail)
    with pytest.raises(RefusesAttachment) as caught:
        _call(runner.run_internal_process_with_evidence, binding, paths)
    assert caught.value is original
