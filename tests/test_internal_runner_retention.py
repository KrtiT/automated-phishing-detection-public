"""Internal catchable failures preserve completed observations without retries."""

import asyncio
import json
from contextlib import contextmanager
from dataclasses import asdict

import pytest
from test_source_runner import inputs as inputs
from test_source_runner import runner as runner


def _failure(paths):
    path = paths.attempt / "failure-progress.json"
    assert path.is_file(), "missing durable private failure progress"
    return json.loads(path.read_bytes())


def test_late_primary_failure_retains_prefix_and_actual_counts(runner, inputs):
    binding, paths, session, unused = inputs
    session.primary.scorer.fail_at = 2
    with pytest.raises(runner.SourceExecutionError):
        runner._run_bound_internal(binding, paths)
    progress = _failure(paths)
    assert progress["status"] == "failed"
    assert progress["producer"]["inference_counts"] == asdict(
        session.primary.scorer.counts
    )
    assert progress["producer"]["started_primary_position"] == 3
    assert len(progress["producer"]["completed_primary_rows"]) == 2
    assert not paths.public_summary.exists()
    assert (paths.attempt / "outcome.json").is_file()


def _fail_third_row(runner, monkeypatch, interruption):
    score = runner.evaluation_producer._score_row

    def fail(record, session, thresholds, position):
        if position == 3:
            raise interruption
        return score(record, session, thresholds, position)

    monkeypatch.setattr(runner.evaluation_producer, "_score_row", fail)


def _break_cleanup(runner, monkeypatch):
    original = runner.open_bound_evaluation_session

    @contextmanager
    def broken(*args):
        with original(*args) as session:
            try:
                yield session
            finally:
                raise ValueError("private-cleanup-canary")

    monkeypatch.setattr(runner, "open_bound_evaluation_session", broken)


@pytest.mark.parametrize(
    "kind", [asyncio.CancelledError, KeyboardInterrupt, SystemExit]
)
@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_interruption_survives_cleanup_and_retains_progress(
    runner, inputs, monkeypatch, kind, cleanup_failure
):
    binding, paths, unused_session, unused_events = inputs
    interruption = kind("private-interruption-canary")
    _fail_third_row(runner, monkeypatch, interruption)
    if cleanup_failure:
        _break_cleanup(runner, monkeypatch)
    with pytest.raises(kind) as caught:
        runner._run_bound_internal(binding, paths)
    assert caught.value is interruption
    assert "private-interruption-canary" not in str(caught.value)
    progress = _failure(paths)
    assert len(progress["producer"]["completed_primary_rows"]) == 2
    assert progress["cleanup_failed"] is cleanup_failure
    assert not paths.public_summary.exists()
    assert (paths.attempt / "outcome.json").is_file()


def test_complete_scoring_survives_session_teardown_failure(
    runner, inputs, monkeypatch
):
    binding, paths, unused_session, unused_events = inputs
    _break_cleanup(runner, monkeypatch)
    with pytest.raises(runner.SourceExecutionError):
        runner._run_bound_internal(binding, paths)
    progress = _failure(paths)
    scientific = paths.attempt / "scientific-checkpoints"
    assert (scientific / "predictions.jsonl").is_file()
    assert (scientific / "secondary.json").is_file()
    assert (scientific / "completion.json").is_file()
    assert progress["cleanup_failed"] is True
    assert not paths.public_summary.exists()


def test_post_session_failure_is_not_mislabeled_as_cleanup(runner, inputs, monkeypatch):
    binding, paths, unused_session, unused_events = inputs
    original = runner.recheck_binding
    checks = []

    def final_failure(*args):
        checks.append(True)
        if len(checks) == 2:
            raise ValueError("private-final-binding-canary")
        return original(*args)

    monkeypatch.setattr(runner, "recheck_binding", final_failure)
    with pytest.raises(runner.SourceExecutionError):
        runner._run_bound_internal(binding, paths)
    progress = _failure(paths)
    assert progress["stage"] == "final_binding"
    assert progress["cleanup_failed"] is False


def test_success_keeps_complete_checkpoint_inventory_and_five_outputs(runner, inputs):
    binding, paths, unused_session, unused_events = inputs
    runner._run_bound_internal(binding, paths)
    scientific = paths.attempt / "scientific-checkpoints"
    assert len(tuple(scientific.iterdir())) == 21
    private = paths.attempt / "evidence"
    assert len(tuple(private.iterdir())) == 5
    assert all(
        (scientific / path.name).read_bytes() == path.read_bytes()
        for path in private.iterdir()
    )
    assert not (paths.attempt / "failure-progress.json").exists()
