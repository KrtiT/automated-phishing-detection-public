"""Closed retained-preparation entry; supplied digests confer no access authority."""

from pathlib import Path

from . import source_runner
from ._exception_cleanup import CleanupStack
from ._prepared_failure_context import carry_failure_context
from ._prepared_internal_process import _worker_command, retain_held_worker
from ._prepared_internal_records import PreparedInternalRunPaths, require_paths
from .execution_preflight import bind_execution
from .internal_process_handoff import ObservedInternalCompletion, retain_worker_failure
from .owned_worker import observe_worker
from .retained_study_preparation import RestoredStudyPreparation
from .source_completion import verify_prepared_internal_completion_snapshot
from .source_runner import SourceExecutionError
from .study_preparation_context import bound_preparation_context
from .study_preparation_transport import hold_study_preparation

__all__ = [
    "PreparedInternalRunPaths",
    "SourceExecutionError",
    "run_prepared_internal_evaluation",
    "run_prepared_internal_process_with_evidence",
]


def _run_bound_prepared_internal(binding, paths, preparation):
    """Reuse the sole scientific lifecycle; this private seam grants no access."""
    return source_runner._run_bound_internal(binding, paths, preparation=preparation)


def _run_observed_prepared_internal(binding, paths, *, preparation):
    """Use the observing parent's retained expectation, never child-selected hashes."""
    if type(preparation) is not RestoredStudyPreparation:
        raise SourceExecutionError("invalid_retained_preparation")
    command = _worker_command(
        binding,
        paths,
        reservation_sha256=preparation.reservation_sha256,
        completion_sha256=preparation.completion_sha256,
    )
    stage, observed = "worker_acceptance", None
    try:
        observed = observe_worker(command)
        source_runner._require_successful_worker(observed, command)
        stage = "completion_verification"
        snapshot = verify_prepared_internal_completion_snapshot(
            binding,
            paths,
            preparation=preparation,
            producer_exit_code=observed.exit.exit_code,
        )
        return ObservedInternalCompletion(observed, snapshot)
    except BaseException as error:
        if observed is not None:
            retain_worker_failure(error, observed, binding, stage)
        raise


def _enter_preparation(cleanup, paths, context, reservation, completion):
    identity, unused_source, buffers, unused_profile = context
    context = hold_study_preparation(
        paths.preparation,
        expected_identity=identity,
        expected_reservation_sha256=reservation,
        expected_completion_sha256=completion,
        source_spec_bytes=buffers[source_runner._SOURCE],
        preparation_summary_bytes=buffers[source_runner._PREPARATION],
    )
    cleanup.push(context.__exit__)
    return context.__enter__()


def _run_held(binding, paths, context, reservation, completion, observed):
    require_paths(paths)
    completed, body_error = None, None
    try:
        with CleanupStack() as cleanup:
            preparation = _enter_preparation(
                cleanup, paths, context, reservation, completion
            )
            try:
                completed = (
                    _run_observed_prepared_internal(
                        binding, paths, preparation=preparation
                    )
                    if observed
                    else _run_bound_prepared_internal(binding, paths, preparation)
                )
            except BaseException as error:
                body_error = error
                raise
        return completed
    except BaseException as error:
        carry_failure_context(error, body_error)
        retain_held_worker(error, completed, binding)
        raise


def _run_public(root, revision, contract, paths, reservation, completion, observed):
    binding = bind_execution(
        root, expected_revision=revision, expected_contract_sha256=contract
    )
    if not binding.protected_evaluation_ready:
        raise SourceExecutionError("pre_access_freeze_incomplete")
    context = bound_preparation_context(binding)
    if not context[3].protected_evaluation_ready:
        raise SourceExecutionError("pre_access_freeze_incomplete")
    return _run_held(binding, paths, context, reservation, completion, observed)


def run_prepared_internal_evaluation(
    root: Path,
    *,
    expected_revision: str,
    expected_contract_sha256: str,
    paths: PreparedInternalRunPaths,
    expected_preparation_reservation_sha256: str,
    expected_preparation_completion_sha256: str,
) -> Path:
    """Require both frozen access gates before inspecting supplied paths."""
    return _run_public(
        root,
        expected_revision,
        expected_contract_sha256,
        paths,
        expected_preparation_reservation_sha256,
        expected_preparation_completion_sha256,
        False,
    )


def run_prepared_internal_process_with_evidence(
    root: Path,
    *,
    expected_revision: str,
    expected_contract_sha256: str,
    paths: PreparedInternalRunPaths,
    expected_preparation_reservation_sha256: str,
    expected_preparation_completion_sha256: str,
):
    """The actual preparation parent supplies expected digests, not file claims."""
    return _run_public(
        root,
        expected_revision,
        expected_contract_sha256,
        paths,
        expected_preparation_reservation_sha256,
        expected_preparation_completion_sha256,
        True,
    )
