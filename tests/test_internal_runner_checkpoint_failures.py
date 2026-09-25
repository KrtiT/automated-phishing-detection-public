"""Durable checkpoint ambiguity consumes the attempt without further inference."""

import base64
import json

import pytest
from test_internal_runner_retention import _failure
from test_source_runner import inputs as inputs
from test_source_runner import runner as runner

from automated_phishing_detection import _internal_scientific_io as checkpoint_io


@pytest.mark.parametrize("installed", [False, True])
def test_ambiguous_primary_checkpoint_write_never_starts_secondary(
    runner, inputs, monkeypatch, installed
):
    binding, paths, session, unused = inputs
    original = checkpoint_io.append
    writes = []

    def fail(attempt, identities, confirmed, name, content):
        if name != "primary-scores.jsonl":
            return original(attempt, identities, confirmed, name, content)
        writes.append(content)
        if installed:
            original(attempt, identities, confirmed, name, content)
        raise OSError("private-checkpoint-canary")

    def forbidden(*args, **kwargs):
        pytest.fail("secondary inference after ambiguous primary write")

    monkeypatch.setattr(checkpoint_io, "append", fail)
    monkeypatch.setattr(runner.evaluation_producer, "score_bound_secondary", forbidden)
    with pytest.raises(runner.SourceExecutionError, match="scoring: execution_failed"):
        runner._run_bound_internal(binding, paths)
    _assert_ambiguous(paths, writes, installed)
    assert len(session.primary.scorer.urls) == 4


def _assert_ambiguous(paths, writes, installed):
    assert len(writes) == 1
    progress = _failure(paths)
    pending = progress["checkpoints"]["pending_checkpoint_bytes"]
    assert base64.b64decode(pending["primary-scores.jsonl"]) == writes[0]
    assert progress["producer"]["failed_checkpoint"] == "primary-scores.jsonl"
    assert len(progress["producer"]["completed_primary_rows"]) == 4
    scientific = paths.attempt / "scientific-checkpoints"
    assert (scientific / "primary-scores.jsonl").exists() is installed
    if installed:
        assert (scientific / "primary-scores.jsonl").read_bytes() == writes[0]
    assert not (scientific / "primary-completion.json").exists()
    assert not paths.public_summary.exists()


def test_late_aggregation_failure_keeps_all_complete_scores(
    runner, inputs, monkeypatch
):
    binding, paths, unused_session, unused_events = inputs

    def fail(*args):
        raise ValueError("private-aggregate-canary")

    monkeypatch.setattr(runner, "_secondary", fail)
    with pytest.raises(runner.SourceExecutionError):
        runner._run_bound_internal(binding, paths)
    progress = _failure(paths)
    scientific = paths.attempt / "scientific-checkpoints"
    assert len(tuple(scientific.iterdir())) == 19
    predictions = (scientific / "predictions.jsonl").read_bytes().splitlines()
    assert len(predictions) == 4
    assert all(len(json.loads(line)["secondary_seeds"]) == 5 for line in predictions)
    assert len(progress["producer"]["completed_secondary_columns"]) == 12
    assert progress["cleanup_failed"] is False
    assert not paths.public_summary.exists()


def test_publication_failure_does_not_add_failure_sidecar(runner, inputs, monkeypatch):
    binding, paths, unused_session, unused_events = inputs
    original = runner.publish_completion

    def fail_after_install(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("private-publication-canary")

    def forbidden(*args, **kwargs):
        pytest.fail("replacement failure sidecar after publication started")

    monkeypatch.setattr(runner, "publish_completion", fail_after_install)
    monkeypatch.setattr(runner, "retain_failure_progress", forbidden)
    with pytest.raises(runner.SourceExecutionError, match="publication"):
        runner._run_bound_internal(binding, paths)
    assert paths.public_summary.exists()
    assert not (paths.attempt / "failure-progress.json").exists()
