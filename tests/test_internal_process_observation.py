"""The official parent requires an owned observation before reading completion."""

import asyncio
from types import SimpleNamespace

import pytest
from test_source_runner import inputs as inputs
from test_source_runner import runner as runner

from automated_phishing_detection import execution_preflight, source_completion
from automated_phishing_detection._owned_process_exit import OwnedProcessExit


def _api(runner):
    assert hasattr(runner, "run_internal_process_with_evidence"), (
        "missing observed handoff"
    )
    return runner.run_internal_process_with_evidence


def _call(method, binding, paths):
    return method(
        binding.root,
        expected_revision=binding.revision,
        expected_contract_sha256=binding.contract_sha256,
        paths=paths,
    )


def _ready(runner, binding, monkeypatch):
    monkeypatch.setattr(runner, "bind_execution", lambda *args, **kwargs: binding)
    monkeypatch.setattr(
        execution_preflight.ExecutionBinding,
        "protected_evaluation_ready",
        property(lambda self: True),
    )


def _observer(runner, monkeypatch, exit_result, *, bad_command=False):
    from automated_phishing_detection._process_support import command_hash
    from automated_phishing_detection.owned_worker import WorkerObservation

    observed = []

    def supervise(command):
        digest = "f" * 64 if bad_command else command_hash(command)
        result = WorkerObservation(digest, exit_result, "a" * 64, "b" * 64)
        observed.append((command, result))
        return result

    monkeypatch.setattr(runner, "observe_worker", supervise)
    return observed


@pytest.mark.parametrize("legacy", [False, True])
def test_successful_parent_retains_exact_snapshot_and_observation(
    runner, inputs, monkeypatch, legacy
):
    method = _api(runner)
    binding, paths, unused_session, unused_events = inputs
    _ready(runner, binding, monkeypatch)
    observed = _observer(runner, monkeypatch, OwnedProcessExit(123, True, 0))
    snapshot, reads = SimpleNamespace(public_summary={"verified": True}), []

    def verify(actual_binding, actual_paths, *, producer_exit_code):
        assert actual_binding is binding and actual_paths is paths
        assert producer_exit_code == 0 and len(observed) == 1
        reads.append(True)
        return snapshot

    monkeypatch.setattr(
        source_completion, "verify_internal_completion_snapshot", verify
    )
    result = _call(runner.run_internal_process if legacy else method, binding, paths)
    assert reads == [True]
    if legacy:
        assert result == {"verified": True}
    else:
        assert result.snapshot is snapshot
        assert result.worker is observed[0][1]
    assert type(observed[0][0]) is tuple


@pytest.mark.parametrize(
    "exit_result",
    [
        OwnedProcessExit(123, False, None),
        OwnedProcessExit(123, False, 0),
        OwnedProcessExit(123, True, 17),
        OwnedProcessExit(123, True, -9),
        OwnedProcessExit(123, True, False),
    ],
)
def test_unknown_nonzero_and_noninteger_exit_never_read_completion(
    runner, inputs, monkeypatch, exit_result
):
    method = _api(runner)
    binding, paths, unused_session, unused_events = inputs
    _ready(runner, binding, monkeypatch)
    observed = _observer(runner, monkeypatch, exit_result)

    def forbidden(*args, **kwargs):
        pytest.fail("unaccepted worker reached saved verification")

    monkeypatch.setattr(
        source_completion, "verify_internal_completion_snapshot", forbidden
    )
    with pytest.raises(runner.SourceExecutionError, match="worker_exit"):
        _call(method, binding, paths)
    assert len(observed) == 1


def test_different_observed_command_never_reads_completion(runner, inputs, monkeypatch):
    method = _api(runner)
    binding, paths, unused_session, unused_events = inputs
    _ready(runner, binding, monkeypatch)
    _observer(runner, monkeypatch, OwnedProcessExit(123, True, 0), bad_command=True)

    def forbidden(*args, **kwargs):
        pytest.fail("different command reached saved verification")

    monkeypatch.setattr(
        source_completion, "verify_internal_completion_snapshot", forbidden
    )
    with pytest.raises(runner.SourceExecutionError, match="worker_command"):
        _call(method, binding, paths)


def test_snapshot_parent_remains_closed_before_launch(runner, inputs, monkeypatch):
    method = _api(runner)
    binding, paths, unused_session, unused_events = inputs
    monkeypatch.setattr(runner, "bind_execution", lambda *args, **kwargs: binding)

    def forbidden(*args, **kwargs):
        pytest.fail("closed entry launched worker")

    monkeypatch.setattr(runner, "observe_worker", forbidden)
    with pytest.raises(runner.SourceExecutionError, match="pre_access_freeze"):
        _call(method, binding, paths)


@pytest.mark.parametrize(
    "kind", [asyncio.CancelledError, KeyboardInterrupt, SystemExit]
)
def test_parent_preserves_worker_interruption_without_verification(
    runner, inputs, monkeypatch, kind
):
    method = _api(runner)
    binding, paths, unused_session, unused_events = inputs
    _ready(runner, binding, monkeypatch)
    original = kind()

    def interrupted(*args):
        raise original

    def forbidden(*args, **kwargs):
        pytest.fail("interrupted worker reached saved verification")

    monkeypatch.setattr(runner, "observe_worker", interrupted)
    monkeypatch.setattr(
        source_completion, "verify_internal_completion_snapshot", forbidden
    )
    with pytest.raises(kind) as caught:
        _call(method, binding, paths)
    assert caught.value is original
