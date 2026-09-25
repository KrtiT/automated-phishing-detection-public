"""Retained inputs do not change first-failure or permanent publication semantics."""

import asyncio
import json
import signal

import pytest
from study_preparation_runner_fixtures import (
    inputs,
    preparation_api,
    preparation_case,
    runner,
)
from test_internal_runner_retention import _break_cleanup, _fail_third_row
from test_prepared_internal_process import process_case
from test_prepared_internal_runner import module

from automated_phishing_detection import source_runner

__all__ = ["inputs", "preparation_api", "preparation_case", "runner", "process_case"]


@pytest.mark.parametrize(
    "kind", [KeyboardInterrupt, SystemExit, asyncio.CancelledError]
)
@pytest.mark.parametrize("cleanup_failed", [False, True])
def test_first_body_interruption_and_partial_progress_survive_teardown(
    process_case, monkeypatch, kind, cleanup_failed
):
    case, original = process_case, kind("private failure")
    _fail_third_row(source_runner, monkeypatch, original)
    if cleanup_failed:
        _break_cleanup(source_runner, monkeypatch)
    with pytest.raises(kind) as caught:
        module()._run_bound_prepared_internal(
            case.binding, case.paths, case.preparation
        )
    assert caught.value is original
    progress = json.loads((case.paths.attempt / "failure-progress.json").read_bytes())
    assert len(progress["producer"]["completed_primary_rows"]) == 2
    assert progress["cleanup_failed"] is cleanup_failed
    assert len(case.session.primary.scorer.urls) == 2
    assert not case.paths.public_summary.exists()


def test_publication_failure_never_falls_back_to_failure_finalization(
    process_case, monkeypatch
):
    case, called = process_case, []

    def publish(*args, **kwargs):
        called.append(True)
        raise OSError("publication failed")

    monkeypatch.setattr(source_runner, "publish_completion", publish)
    monkeypatch.setattr(
        source_runner,
        "record_failure",
        lambda *args, **kwargs: pytest.fail("retried finalization"),
    )
    monkeypatch.setattr(
        source_runner,
        "retain_failure_progress",
        lambda *args, **kwargs: pytest.fail("persisted after publishing guard"),
    )
    with pytest.raises(source_runner.SourceExecutionError, match="publication"):
        module()._run_bound_prepared_internal(
            case.binding, case.paths, case.preparation
        )
    assert called == [True]


@pytest.mark.parametrize("function", ["produce_internal_evidence", "_secondary"])
def test_scoring_and_secondary_computation_remain_immediately_interruptible(
    process_case, monkeypatch, function
):
    case, continued = process_case, []

    def interrupted(*args, **kwargs):
        signal.raise_signal(signal.SIGINT)
        continued.append(True)

    owner = (
        source_runner.evaluation_producer
        if function == "produce_internal_evidence"
        else source_runner
    )
    monkeypatch.setattr(owner, function, interrupted)
    with pytest.raises(KeyboardInterrupt):
        module()._run_bound_prepared_internal(
            case.binding, case.paths, case.preparation
        )
    assert continued == []
