"""Closed source-only entry; completed preparation cannot launch scoring.

This stage retains immutable source preparation and necessary population counts.
It changes no hypothesis rule, accepts no models and provides no resume command.
"""

from pathlib import Path

from . import _study_preparation_body as body
from ._exception_cleanup import CleanupStack
from ._study_preparation_records import (
    PreparationState,
    PreparedStudySnapshot,
    StudyPreparationError,
    StudyPreparationPaths,
)
from .execution_preflight import bind_execution, recheck_binding
from .execution_receipt import record_failure
from .study_preparation_retention import retain_study_preparation

__all__ = [
    "PreparedStudySnapshot",
    "StudyPreparationError",
    "StudyPreparationPaths",
    "run_study_preparation",
]


def _interruption(original, later):
    if isinstance(original, Exception) and not isinstance(later, Exception):
        try:
            if type(getattr(original, "preparation_progress", None)) is bytes:
                later.preparation_progress = original.preparation_progress
        except BaseException:
            pass
        return later
    return original


def _progress(state, selected):
    if state.writer is not None:
        try:
            selected.preparation_progress = state.writer.snapshot()
        except BaseException as failure:
            selected = _interruption(selected, failure)
    return selected


def _failure(state, error):
    selected = _progress(
        state,
        error
        if not isinstance(error, Exception)
        else StudyPreparationError("study_preparation_failed"),
    )
    if state.attempt is not None:
        try:
            with body.deferred_io():
                record_failure(
                    state.attempt,
                    stage=state.stage,
                    error_type="preparation_failed"
                    if isinstance(selected, Exception)
                    else "interrupted",
                )
        except BaseException as failure:
            selected = _interruption(selected, failure)
    return selected


def _recheck(binding):
    with body.deferred_io():
        recheck_binding(binding)


def _run_bound_preparation(binding, paths):
    """Exercise invented fixtures privately; this function grants no access."""
    state = PreparationState(binding, paths)
    try:
        _recheck(binding)
        body.preflight(state)
        with CleanupStack() as cleanup:
            with body.deferred_io():
                state.writer = cleanup.enter_context(
                    retain_study_preparation(state.attempt, identity=state.identity)
                )
            internal, domains, suffix = body.prepare_internal(state)
            external = body.prepare_external(state, domains, suffix)
            body.assess(state, internal, external)
            state.stage = "final_binding"
            _recheck(binding)
            payloads = body.complete(state)
            state.stage = "retention_finalization"
        return PreparedStudySnapshot(state.attempt.reservation_sha256, payloads)
    except BaseException as error:
        raise _failure(state, error) from None


def run_study_preparation(
    root: Path,
    *,
    expected_revision: str,
    expected_contract_sha256: str,
    paths: StudyPreparationPaths,
) -> PreparedStudySnapshot:
    """Reject until a separately reviewed preparation-access profile is adopted."""
    binding = bind_execution(
        root,
        expected_revision=expected_revision,
        expected_contract_sha256=expected_contract_sha256,
    )
    if not binding.protected_evaluation_ready:
        raise StudyPreparationError("pre_access_freeze_incomplete")
    profile = body.resolve_external_source_profile(binding)
    if not profile.protected_evaluation_ready:
        raise StudyPreparationError("pre_access_freeze_incomplete")
    return _run_bound_preparation(binding, paths)
