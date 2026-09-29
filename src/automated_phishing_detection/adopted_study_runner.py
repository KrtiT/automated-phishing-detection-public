"""The distinct fresh-root route requires one exact adopted execution envelope."""

import json

from . import _adopted_study_body as body
from . import _adopted_study_records as records
from . import _study_run_body as original
from ._adopted_study_state import AdoptedStudyProgress, adopted_study_failure, attach
from ._exception_cleanup import CleanupStack
from ._study_run_context import StudyRunError, validate_context
from ._study_run_failure import reject
from ._study_run_paths import CELL_NAMES, hold_study_paths
from .study_root_retention import hold_study_root
from .study_runner import ObservedStudyResult

__all__ = ["StudyRunError", "adopted_study_failure", "run_adopted_study"]


def recheck_study_execution(authorization):
    from .study_execution import recheck_study_execution as recheck

    return recheck(authorization)


def _reserve(state, cleanup):
    state.stage = "output_preflight"
    state.outputs = original.enter(cleanup, hold_study_paths(state.paths))
    identity = records.study_identity(state.authorization)
    state.stage = "reservation"
    original.files.deferred(state.reserve, identity)
    state.stage = "root_retention"
    state.writer = original.enter(
        cleanup,
        hold_study_root(
            state.attempt,
            state.paths.public_summary,
            expected_identity=identity,
        ),
    )
    state.writer.append("study-intent.json", state.intent)


async def _work(state, cleanup):
    profile = json.loads(state.authorization.profile_bytes)
    if profile.get("profile_id") == "study-urlnorm-profile-v1":
        from ._study_urlnorm_preparation import prepare

        prepare(state, cleanup)
    else:
        original.prepare(state, cleanup)
    if not state.held:
        state.admissions.barrier_retained(
            state.preparation,
            dict(state.writer.payloads)["prediction-barrier.json"],
        )
        original.files.deferred(recheck_study_execution, state.authorization)
        body.sources(state, cleanup)
        await body.cells(state)
    state.stage = "root_publication"
    body.accounting(state, "whole_study_hold" if state.held else "matrix_accepted")
    state.outputs.check(() if state.held else CELL_NAMES)
    original.files.deferred(recheck_study_execution, state.authorization)
    snapshot = body.publish(state)
    state.stage = "root_finalization"
    return ObservedStudyResult(
        state.preparation, state.sources, state.accepted, state.cells(), snapshot
    )


async def _run(state):
    with CleanupStack() as cleanup:
        try:
            validate_context(state.binding, state.profile, state.paths, state.deadlines)
            original.files.deferred(recheck_study_execution, state.authorization)
            cleanup.callback(
                original.files.deferred, recheck_study_execution, state.authorization
            )
            _reserve(state, cleanup)
            return await _work(state, cleanup)
        except BaseException as error:
            body.failed_accounting(state, error)
            raise


async def _run_adopted_bound(authorization):
    state = AdoptedStudyProgress(authorization)
    try:
        return await _run(state)
    except BaseException as error:
        try:
            reject(state, error)
        except BaseException as selected:
            raise attach(selected, state) from None


async def run_adopted_study(
    root, *, expected_revision, envelope_path, expected_envelope_sha256
):
    from .study_execution import bind_study_execution

    authorization = bind_study_execution(
        root,
        expected_revision=expected_revision,
        envelope_path=envelope_path,
        expected_envelope_sha256=expected_envelope_sha256,
    )
    return await _run_adopted_bound(authorization)
