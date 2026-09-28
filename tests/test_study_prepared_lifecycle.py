"""Prepared source adapters propagate supplement checks into original workers."""

import pytest
from prepared_external_fixtures import prepared_case as prepared_case
from study_execution_fixtures import bind
from study_execution_fixtures import execution_case as execution_case
from study_lifecycle_fixtures import admitted, late_check
from study_preparation_runner_fixtures import inputs as inputs
from study_preparation_runner_fixtures import preparation_api as preparation_api
from study_preparation_runner_fixtures import preparation_case as preparation_case
from study_preparation_runner_fixtures import runner as runner
from test_prepared_internal_process import process_case as process_case

from automated_phishing_detection import _prepared_external_runtime as external
from automated_phishing_detection import external_source_runner as worker
from automated_phishing_detection import prepared_internal_runner as internal
from automated_phishing_detection.internal_handoff_transport import (
    retain_internal_handoff,
)


@pytest.mark.parametrize("boundary", ["envelope", "channel"])
def test_held_internal_propagates_check_through_prepared_body(
    execution_case, process_case, boundary
):
    case = process_case
    context = (case.identity, case.source, case.buffers, case.profile)
    with admitted(bind(execution_case)) as (held, state):
        check = late_check(held, state, case.events, "exit", boundary)
        with pytest.raises(internal.SourceExecutionError, match="final_binding"):
            internal._run_held(
                case.binding,
                case.paths,
                context,
                case.preparation.reservation_sha256,
                case.preparation.completion_sha256,
                False,
                lifecycle_check=check,
            )
    assert len(case.session.primary.scorer.urls) == 4
    assert (case.paths.attempt / "failure-progress.json").exists()
    assert not case.paths.public_summary.exists()


@pytest.mark.parametrize("boundary", ["envelope", "channel"])
def test_held_external_propagates_check_through_prepared_body(
    execution_case, prepared_case, boundary
):
    case = prepared_case
    with admitted(bind(execution_case)) as (held, state):
        check = late_check(held, state, case.events, "session_closed", boundary)
        with retain_internal_handoff(case.handoff) as transport:
            with pytest.raises(
                worker.ExternalSourceExecutionError, match="final_binding"
            ):
                external.run_held_preparation(
                    case.binding,
                    case.paths,
                    transport.directory,
                    transport.expected_handoff_sha256,
                    case.preparation.reservation_sha256,
                    case.preparation.completion_sha256,
                    lifecycle_check=check,
                )
    assert len(case.session.evaluation.primary.scorer.urls) == 1
    assert (case.paths.attempt / "external-failure.json").exists()
    assert not case.paths.public_summary.exists()
