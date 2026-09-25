"""One fresh preparation, one source pair and the unchanged sequential125 cells."""

from hashlib import sha256

from . import _study_preparation_files as files
from . import _study_run_records as records
from ._prepared_external_runtime import held_preparation
from ._study_run_context import require
from ._study_run_failure import select
from ._study_run_paths import CELL_NAMES, cell_paths
from .operational_cell_runner import _run_bound_cell
from .operational_input_transport import retain_operational_root_inputs
from .operational_inputs import build_accepted_inputs
from .operational_schedule import planned_cells
from .prepared_external_process import _run_observed_prepared_sources
from .study_operational_records import retain_accepted_cell
from .study_preparation_runner import _run_bound_preparation
from .study_reduction import reduce_accepted_study


def enter(cleanup, manager):
    cleanup.push(manager.__exit__)
    return manager.__enter__()


def prepare(state, cleanup):
    state.stage = "preparation"
    state.fresh = _run_bound_preparation(state.binding, state.paths.preparation)
    completion = sha256(state.fresh.payload("preparation-complete.json")).hexdigest()
    state.stage = "preparation_retention"
    state.preparation = enter(
        cleanup,
        held_preparation(
            state.binding,
            state.paths.external,
            state.fresh.reservation_sha256,
            completion,
        ),
    )
    require(state.preparation.payloads == state.fresh.payloads)
    require(state.preparation.reservation_sha256 == state.fresh.reservation_sha256)
    require(state.preparation.completion_sha256 == completion)
    state.outputs.check(())
    state.stage = "prediction_barrier"
    barrier, state.held = records.prediction_barrier(
        state.preparation, execution=state.execution
    )
    state.writer.append("prediction-barrier.json", barrier)


def sources(state, cleanup):
    state.outputs.check(())
    state.stage, state.sources_started = "sources", True
    state.sources = _run_observed_prepared_sources(
        state.binding, state.paths.internal, state.paths.external, state.preparation
    )
    state.accepted = build_accepted_inputs(
        state.sources.internal,
        state.sources.external,
        binding=state.binding,
        root_reservation_sha256=state.attempt.reservation_sha256,
        operational_profile_sha256=state.profile.profile_sha256,
    )
    state.writer.append(
        "source-results.json",
        records.source_results(state.accepted, execution=state.execution),
    )
    state.stage = "accepted_inputs"
    enter(
        cleanup,
        retain_operational_root_inputs(
            state.paths.accepted_inputs_directory,
            accepted_inputs=state.accepted.metadata_bytes,
        ),
    )


async def cells(state):
    for cell in planned_cells():
        state.outputs.check(CELL_NAMES[: 3 * len(state.completed)])
        state.current, state.stage = cell, "cell_execution"
        state.returned = await _run_bound_cell(
            state.binding,
            state.profile,
            state.accepted,
            cell,
            paths=cell_paths(state.paths, cell.ordinal),
            artifacts=state.paths.internal.artifacts,
            deadlines=state.deadlines,
        )
        state.stage = "cell_compaction"
        retained = retain_accepted_cell(state.returned, accepted=state.accepted)
        files.deferred(state.commit_cell, retained)
    state.outputs.check(CELL_NAMES)
    state.stage = "reduction"
    state.reduced = reduce_accepted_study(state.accepted, state.cells())


def accounting(state, status, error=None):
    if state.accounting_started or state.writer is None:
        return
    state.accounting_started = True
    internal, external = state.source_statuses(error)
    content = records.study_accounting(
        state.cells(error),
        execution=state.execution,
        stage=state.stage,
        status=status,
        internal_status=internal,
        external_status=external,
    )
    state.writer.append("study-accounting.json", content)


def failed_accounting(state, error):
    state.remember(error)
    try:
        accounting(state, "failed", error)
    except BaseException as later:
        selected = select(error, later)
        if selected is not error:
            raise selected from None


def publish(state):
    extra = (
        {}
        if state.reduced is None
        else {
            "operational-summary.json": state.reduced.operational_bytes,
            "study-evidence.json": state.reduced.study_bytes,
        }
    )
    public = records.root_public_summary(
        execution=state.execution,
        checkpoints=state.writer.payloads,
        reduced=state.reduced,
    )
    return state.writer.complete(extra_outputs=extra, public_summary=public)
