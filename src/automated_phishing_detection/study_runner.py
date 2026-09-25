"""Closed whole-study entry; one-pass fixture composition never authorizes access."""

from dataclasses import dataclass, field

from . import _study_preparation_files as files
from . import _study_run_body as body
from ._exception_cleanup import CleanupStack
from ._external_source_profile import resolve_external_source_profile
from ._operational_profile import resolve_operational_profile
from ._study_run_context import StudyRunError, StudyRunPaths, validate_context
from ._study_run_failure import reject
from ._study_run_paths import CELL_NAMES, hold_study_paths
from ._study_run_state import StudyProgress, StudyRunFailure
from .execution_preflight import bind_execution, recheck_binding
from .study_root_retention import hold_study_root

__all__ = [
    "StudyRunError",
    "StudyRunPaths",
    "StudyRunFailure",
    "ObservedStudyResult",
    "run_study",
    "study_failure",
]


@dataclass(frozen=True)
class ObservedStudyResult:
    preparation: object = field(repr=False)
    sources: object = field(repr=False)
    accepted: object = field(repr=False)
    cells: tuple = field(repr=False)
    snapshot: object = field(repr=False)


def study_failure(error):
    """Recover actual private context through Python 3.10 cancellation wrapping."""
    seen = set()
    while isinstance(error, BaseException) and id(error) not in seen:
        seen.add(id(error))
        retained = getattr(error, "study_failure", None)
        if type(retained) is StudyRunFailure:
            return retained
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return None


def _reserve(state, cleanup):
    state.stage = "output_preflight"
    state.outputs = body.enter(cleanup, hold_study_paths(state.paths))
    identity = body.records.study_identity(state.binding, state.profile)
    state.stage = "reservation"
    files.deferred(state.reserve, identity)
    state.stage = "root_retention"
    state.writer = body.enter(
        cleanup,
        hold_study_root(
            state.attempt, state.paths.public_summary, expected_identity=identity
        ),
    )
    state.writer.append(
        "study-intent.json",
        body.records.study_intent(
            state.binding, state.profile, state.attempt, state.deadlines
        ),
    )


async def _work(state, cleanup):
    body.prepare(state, cleanup)
    if not state.held:
        body.sources(state, cleanup)
        await body.cells(state)
    state.stage = "root_publication"
    body.accounting(state, "whole_study_hold" if state.held else "matrix_accepted")
    state.outputs.check(() if state.held else CELL_NAMES)
    files.deferred(recheck_binding, state.binding)
    snapshot = body.publish(state)
    state.stage = "root_finalization"
    return ObservedStudyResult(
        state.preparation, state.sources, state.accepted, state.cells(), snapshot
    )


async def _run(state):
    with CleanupStack() as cleanup:
        try:
            validate_context(state.binding, state.profile, state.paths, state.deadlines)
            state.deadlines = state.deadlines.copy()
            files.deferred(recheck_binding, state.binding)
            cleanup.callback(files.deferred, recheck_binding, state.binding)
            _reserve(state, cleanup)
            return await _work(state, cleanup)
        except BaseException as error:
            body.failed_accounting(state, error)
            raise


async def _run_bound_study(binding, profile, *, paths, deadlines):
    state = StudyProgress(binding, profile, paths, deadlines)
    try:
        return await _run(state)
    except BaseException as error:
        reject(state, error)


async def run_study(
    root,
    *,
    expected_revision,
    expected_contract_sha256,
    expected_operational_profile_sha256,
    paths,
):
    binding = bind_execution(
        root,
        expected_revision=expected_revision,
        expected_contract_sha256=expected_contract_sha256,
    )
    if not binding.protected_evaluation_ready:
        raise StudyRunError("pre_access_freeze_incomplete")
    external = files.deferred(resolve_external_source_profile, binding)
    if not external.protected_evaluation_ready:
        raise StudyRunError("external_profile_freeze_incomplete")
    profile = files.deferred(resolve_operational_profile, binding)
    if not profile.protected_evaluation_ready:
        raise StudyRunError("operational_profile_freeze_incomplete")
    if profile.profile_sha256 != expected_operational_profile_sha256:
        raise StudyRunError("operational_profile_mismatch")
    return await _run_bound_study(
        binding,
        profile,
        paths=paths,
        deadlines=profile.projection()["protective_deadlines_seconds"],
    )
