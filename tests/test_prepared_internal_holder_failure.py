"""Late holder errors retain the real observation without accepting the result."""

from contextlib import contextmanager

import pytest
from study_preparation_runner_fixtures import (
    inputs,
    preparation_api,
    preparation_case,
    runner,
)
from test_internal_external_handoff import observed_worker
from test_prepared_internal_process import _held_result, process_case
from test_prepared_internal_runner import module

from automated_phishing_detection.internal_process_handoff import (
    ObservedInternalCompletion,
    ObservedInternalFailure,
)

__all__ = [
    "inputs",
    "preparation_api",
    "preparation_case",
    "runner",
    "process_case",
    "observed_worker",
]


def failed_holder(api, case, monkeypatch, cleanup):
    @contextmanager
    def hold(*args, **kwargs):
        try:
            yield case.preparation
        finally:
            raise cleanup

    monkeypatch.setattr(api, "hold_study_preparation", hold)


@pytest.mark.parametrize("kind", [OSError, KeyboardInterrupt, SystemExit])
def test_successful_observation_survives_late_holder_failure(
    process_case, observed_worker, monkeypatch, kind
):
    api, case, cleanup = module(), process_case, kind("holder failed")
    completed = ObservedInternalCompletion(observed_worker, object())
    failed_holder(api, case, monkeypatch, cleanup)
    monkeypatch.setattr(
        api, "_run_observed_prepared_internal", lambda *args, **kwargs: completed
    )
    with pytest.raises(kind) as caught:
        _held_result(api, case)
    assert caught.value is cleanup
    assert caught.value.worker_failure.worker is observed_worker
    assert caught.value.worker_failure.binding is case.binding
    assert caught.value.worker_failure.stage == "preparation_finalization"


def raise_body(error):
    def fail(*args, **kwargs):
        raise error

    return fail


@pytest.mark.parametrize("kind", [OSError, KeyboardInterrupt, SystemExit])
def test_replacement_cleanup_error_keeps_existing_private_failure_context(
    process_case, observed_worker, monkeypatch, kind
):
    api, case, cleanup = module(), process_case, kind("holder failed")
    original = ValueError("verification failed")
    association = ObservedInternalFailure(
        observed_worker, case.binding, "completion_verification"
    )
    original.worker_failure = association
    original.progress = b"already retained bytes"
    cleanup.preparation_progress = b"later holder bytes"
    failed_holder(api, case, monkeypatch, cleanup)
    monkeypatch.setattr(api, "_run_observed_prepared_internal", raise_body(original))
    with pytest.raises(kind) as caught:
        _held_result(api, case)
    assert caught.value is cleanup
    assert caught.value.worker_failure is association
    assert caught.value.progress is original.progress
    assert caught.value.preparation_progress == b"later holder bytes"


@pytest.mark.parametrize("kind", [OSError, KeyboardInterrupt, SystemExit])
def test_original_interruption_stays_first_with_its_association(
    process_case, observed_worker, monkeypatch, kind
):
    api, case = module(), process_case
    original = KeyboardInterrupt("first interruption")
    original.worker_failure = ObservedInternalFailure(
        observed_worker, case.binding, "completion_verification"
    )
    failed_holder(api, case, monkeypatch, kind("holder failed"))
    monkeypatch.setattr(api, "_run_observed_prepared_internal", raise_body(original))
    with pytest.raises(KeyboardInterrupt) as caught:
        _held_result(api, case)
    assert caught.value is original
    assert caught.value.worker_failure.worker is observed_worker
