"""Private one-segment orchestration; public launch still requires its supervisor."""

import json

from . import _study_preparation_files as files
from . import _study_series_child_commands as commands
from . import _study_series_runner_body as body
from . import _study_series_runner_records as records
from ._exception_cleanup import CleanupStack
from ._study_run_body import enter
from ._study_series_runner_failure import failed_accounting, reject
from ._study_series_runner_paths import cell_names, check_outputs, hold_outputs
from ._study_series_runner_state import (
    SeriesProgress,
    SeriesRunError,
    SeriesRunFailure,
    SeriesRunResult,
    series_run_failure,
)
from .operational_schedule import planned_cells
from .study_series_cell_runner import _run_series_cell
from .study_series_execution import recheck_series_public_execution
from .study_series_history_index import validate_series_history_index

__all__ = [
    "SeriesRunError",
    "SeriesRunFailure",
    "SeriesRunResult",
    "series_run_failure",
]


def require_final_policy(public):
    if json.loads(public.policy_bytes)["status"] != "specified_for_explicit_adoption":
        raise SeriesRunError("series_execution_not_adopted")


def check_public(state):
    recheck_series_public_execution(state.public)
    require_final_policy(state.public)
    state.lifecycle_check()


def check(state):
    files.deferred(check_public, state)
    files.deferred(state.history.check)


def validate_history(state):
    history, public = state.history, state.public
    index = validate_series_history_index(
        history.index.index_bytes,
        public.profile_bytes,
        expected_index_sha256=state.profile["history"]["index_sha256"],
        expected_profile_sha256=public.profile_sha256,
    )
    files.require(index == history.index)
    files.require(type(history.historical_prefix) is tuple)
    files.require(len(history.historical_prefix) == len(index.accepted_ordinals))
    files.require(
        type(history.origin_metadata_bytes) is bytes and callable(history.check)
    )
    check(state)


async def cells(state):
    names = cell_names(state.profile)
    for cell in planned_cells()[state.profile["segment"]["start_ordinal"] - 1 :]:
        state.stage = "cell_execution"
        check(state)
        check_outputs(state.outputs, names, names[: 3 * len(state.completed)])
        result = await _run_series_cell(
            state.public,
            state.metadata_bytes,
            state.history.internal_snapshot,
            state.history.external_snapshot,
            cell,
            admissions=state.ledger,
        )
        state.completed.append(result)
    check_outputs(state.outputs, names, names)
    check(state)
    state.reduced = records.reduce(state)


def publish(state):
    check(state)
    state.stage = "segment_finalization"
    records.account(state, "complete")
    records.append_accounting(state)
    state.stage = "segment_publication"
    records.publication(state, "segment")
    check(state)
    records.publication(state, "series")
    state.stage = "segment_finalization"


async def _run(state, hold_history):
    with CleanupStack() as cleanup:
        try:
            files.deferred(check_public, state)
            cleanup.callback(files.deferred, check_public, state)
            state.history = enter(cleanup, hold_history(state.public))
            validate_history(state)
            cleanup.callback(files.deferred, state.history.check)
            state.stage = "output_preflight"
            state.outputs = enter(cleanup, hold_outputs(state.profile, state.history))
            body.reserve(state, cleanup)
            body.prepare(state, cleanup)
            await cells(state)
            publish(state)
        except BaseException as error:
            failed_accounting(state, error)
            raise


async def _run_series_bound(public, *, hold_history, lifecycle_check):
    profile = commands._profile(public)
    require_final_policy(public)
    files.require(callable(hold_history) and callable(lifecycle_check))
    state = SeriesProgress(public, profile, lifecycle_check)
    try:
        await _run(state, hold_history)
    except BaseException as error:
        reject(state, error)
    return SeriesRunResult(
        state.series_attempt,
        state.segment_attempt,
        state.segment_writer.candidate,
        state.series_writer.candidate,
        state.segment_accounting_bytes,
        state.series_accounting_bytes,
        tuple(state.completed),
        state.reduced,
    )
