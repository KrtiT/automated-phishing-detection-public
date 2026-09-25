"""Preparation joins fail before numerical ownership or scientific reservation."""

from dataclasses import replace

import pytest
from prepared_external_fixtures import (
    inputs,
    preparation_api,
    preparation_case,
    prepared_case,
    runner,
)

from automated_phishing_detection import external_source_runner as worker

__all__ = ["inputs", "preparation_api", "preparation_case", "prepared_case", "runner"]


@pytest.mark.parametrize("field", ["reservation_sha256", "completion_sha256"])
def test_independent_preparation_identity_rejects_another_handoff(prepared_case, field):
    case = prepared_case
    preparation = replace(case.preparation, **{field: "0" * 64})
    with pytest.raises(worker.ExternalSourceExecutionError):
        worker._run_bound_prepared_external(
            case.binding, case.paths, handoff=case.handoff, preparation=preparation
        )
    assert not case.paths.attempt.exists()
    assert not case.events


@pytest.mark.parametrize("field", ["attempt", "public_summary"])
def test_scoring_outputs_cannot_modify_held_preparation(prepared_case, field):
    case = prepared_case
    paths = replace(case.paths, **{field: case.paths.preparation / "new-output"})
    with pytest.raises(worker.ExternalSourceExecutionError):
        worker._run_bound_prepared_external(
            case.binding, paths, handoff=case.handoff, preparation=case.preparation
        )
    assert not (case.paths.preparation / "new-output").exists()
    assert not case.events


def test_different_public_binding_fails_before_reservation(prepared_case):
    case = prepared_case
    binding = replace(case.binding, runtime_json='{"other":true}')
    with pytest.raises(worker.ExternalSourceExecutionError):
        worker._run_bound_prepared_external(
            binding, case.paths, handoff=case.handoff, preparation=case.preparation
        )
    assert not case.paths.attempt.exists()


def test_untyped_preparation_is_not_a_worker_input(prepared_case):
    case = prepared_case
    with pytest.raises(worker.ExternalSourceExecutionError):
        worker._run_bound_prepared_external(
            case.binding, case.paths, handoff=case.handoff, preparation=object()
        )
    assert not case.paths.attempt.exists()
