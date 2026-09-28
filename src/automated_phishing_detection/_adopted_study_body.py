"""Thin adopted orchestration calls the original scientific owners exactly once."""

from . import _adopted_study_records as records
from . import _study_run_body as original
from ._study_run_failure import select
from ._study_run_paths import CELL_NAMES, cell_paths


def sources(state, cleanup):
    from ._study_authorized_sources import run_adopted_sources

    state.outputs.check(())
    state.stage, state.sources_started = "sources", True
    state.sources = run_adopted_sources(
        state.authorization, state.preparation, state.admissions
    )
    state.accepted = original.build_accepted_inputs(
        state.sources.internal,
        state.sources.external,
        binding=state.binding,
        root_reservation_sha256=state.attempt.reservation_sha256,
        operational_profile_sha256=state.profile.profile_sha256,
    )
    content = original.records.source_results(state.accepted, execution=state.execution)
    state.writer.append("source-results.json", content)
    state.stage = "accepted_inputs"
    original.enter(
        cleanup,
        original.retain_operational_root_inputs(
            state.paths.accepted_inputs_directory,
            accepted_inputs=state.accepted.metadata_bytes,
        ),
    )
    state.admissions.inputs_retained(state.accepted, content)


async def cells(state):
    from ._study_authorized_cell import run_adopted_cell

    for cell in original.planned_cells():
        state.outputs.check(CELL_NAMES[: 3 * len(state.completed)])
        state.current, state.stage = cell, "cell_execution"
        state.admissions.cell_started(cell)
        state.returned = await run_adopted_cell(
            state.authorization,
            state.accepted,
            cell,
            paths=cell_paths(state.paths, cell.ordinal),
            admissions=state.admissions,
        )
        state.stage = "cell_compaction"
        retained = original.retain_accepted_cell(
            state.returned, accepted=state.accepted
        )
        state.admissions.cell_accepted(
            cell,
            retained,
            pair_intent_bytes=dict(state.returned.snapshot.payloads)[
                "attempt/process-pair-intent.json"
            ],
        )
        original.files.deferred(state.commit_cell, retained)
    state.outputs.check(CELL_NAMES)
    state.stage = "reduction"
    state.reduced = original.reduce_accepted_study(state.accepted, state.cells())


def accounting(state, status, error=None):
    if state.accounting_started or state.writer is None:
        return
    state.accounting_started = True
    state.writer.append(
        "study-accounting.json", records.study_accounting(state, status, error)
    )


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
    public = records.public_summary(
        state.adopted_execution, state.writer.payloads, state.reduced
    )
    return state.writer.complete(extra_outputs=extra, public_summary=public)
