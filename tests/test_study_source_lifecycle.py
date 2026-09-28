"""Late supplement or live-channel loss cannot publish source completion."""

import asyncio
import json

import pytest
from study_execution_fixtures import bind
from study_execution_fixtures import execution_case as execution_case
from study_lifecycle_fixtures import admitted, invalidate, late_check
from test_external_source_runner import runner_api as runner_api
from test_external_source_runner import runner_case as runner_case
from test_source_runner import inputs as inputs
from test_source_runner import runner as runner


@pytest.mark.parametrize("boundary", ["envelope", "channel"])
def test_internal_late_invalidation_retains_progress_without_publication(
    execution_case, runner, inputs, boundary
):
    binding, paths, session, events = inputs
    with admitted(bind(execution_case)) as (held, state):
        check = late_check(held, state, events, "exit", boundary)
        with pytest.raises(runner.SourceExecutionError, match="final_binding"):
            runner._run_bound_internal(binding, paths, lifecycle_check=check)
    assert len(session.primary.scorer.urls) == 4
    progress = json.loads((paths.attempt / "failure-progress.json").read_bytes())
    assert len(progress["producer"]["completed_primary_rows"]) == 4
    assert progress["stage"] == "final_binding"
    assert not paths.public_summary.exists()
    assert not (paths.attempt / "evidence").exists()


@pytest.mark.parametrize("boundary", ["envelope", "channel"])
def test_external_late_invalidation_retains_progress_without_publication(
    execution_case, runner_api, runner_case, boundary
):
    case = runner_case
    run_state = runner_api.body.ExternalRun(case.binding, case.paths, case.handoff)
    with admitted(bind(execution_case)) as (held, state):
        check = late_check(held, state, case.events, "session_closed", boundary)
        with pytest.raises(
            runner_api.ExternalSourceExecutionError, match="final_binding"
        ):
            runner_api._run(run_state, lifecycle_check=check)
    assert len(case.session.evaluation.primary.scorer.urls) == 5
    failure = json.loads((case.paths.attempt / "external-failure.json").read_bytes())
    assert failure["stage"] == "final_binding"
    assert run_state.failures.produced is not None
    assert run_state.failures.session_closed is True
    assert not case.paths.public_summary.exists()
    assert not (case.paths.attempt / "evidence").exists()


@pytest.mark.parametrize("boundary", ["envelope", "channel"])
def test_internal_invalid_context_stops_before_reservation_or_models(
    execution_case, runner, inputs, boundary
):
    from automated_phishing_detection._study_child_context import recheck_held_child

    binding, paths, session, events = inputs
    with admitted(bind(execution_case)) as (held, state):
        invalidate(held, state, boundary)
        with pytest.raises(runner.SourceExecutionError, match="public_preflight"):
            runner._run_bound_internal(
                binding, paths, lifecycle_check=lambda: recheck_held_child(held)
            )
    assert not paths.attempt.exists()
    assert session.primary.scorer.urls == []
    assert "enter" not in events


@pytest.mark.parametrize(
    "kind", [KeyboardInterrupt, SystemExit, asyncio.CancelledError]
)
def test_internal_lifecycle_interruption_preserves_identity_and_progress(
    runner, inputs, kind
):
    binding, paths, session, events = inputs
    original = kind("invented interruption")

    def check():
        if "exit" in events:
            raise original

    with pytest.raises(kind) as caught:
        runner._run_bound_internal(binding, paths, lifecycle_check=check)
    assert caught.value is original
    progress = json.loads((paths.attempt / "failure-progress.json").read_bytes())
    assert len(progress["producer"]["completed_primary_rows"]) == 4
    assert len(session.primary.scorer.urls) == 4
    assert not paths.public_summary.exists()


@pytest.mark.parametrize(
    "kind", [KeyboardInterrupt, SystemExit, asyncio.CancelledError]
)
def test_external_lifecycle_interruption_preserves_identity_and_progress(
    runner_api, runner_case, kind
):
    case, original = runner_case, kind("invented interruption")
    state = runner_api.body.ExternalRun(case.binding, case.paths, case.handoff)

    def check():
        if "session_closed" in case.events:
            raise original

    with pytest.raises(kind) as caught:
        runner_api._run(state, lifecycle_check=check)
    assert caught.value is original
    assert (
        caught.value.progress
        == (case.paths.attempt / "external-failure.json").read_bytes()
    )
    assert len(case.session.evaluation.primary.scorer.urls) == 5
    assert not case.paths.public_summary.exists()
