"""Private fixed-next-cell composition; the root retains all access/session gates."""

from . import _study_preparation_files as files
from . import _study_series_cell_runner_context as context
from . import operational_cell_runner as original
from ._exception_cleanup import CleanupStack
from ._operational_attempt_io import held_attempt_writer
from ._study_series_cell_runner_state import SeriesCellProgress, reject
from ._study_series_child_commands import series_child_command
from .operational_cell_inputs import bind_cell_descriptor
from .operational_input_transport import retain_operational_cell_inputs
from .study_series_cell_acceptance import series_cell_identity
from .study_series_cell_completion import hold_series_cell
from .study_series_execution import recheck_series_public_execution
from .study_series_process import observe_series_operational_children


def _inputs(state, cleanup):
    state.binding_bytes = bind_cell_descriptor(
        state.selected.descriptor_bytes,
        cell_reservation_sha256=state.attempt.reservation_sha256,
    )
    original._enter(
        cleanup,
        retain_operational_cell_inputs(
            state.paths.cell_input_directory,
            descriptor=state.selected.descriptor_bytes,
            binding=state.binding_bytes,
            manifest=state.selected.manifest_bytes,
        ),
    )
    held = original._enter(
        cleanup,
        context.storage.hold(
            state.paths.accepted_inputs_directory, state.paths.cell_input_directory
        ),
    )
    return context.restore(state, held)


def _prepare(state, cleanup):
    state.stage = "selection"
    state.selected = context.select(state)
    identity = series_cell_identity(
        state.metadata_bytes,
        state.selected.descriptor_bytes,
        expected_metadata_sha256=context.digest(state.metadata_bytes),
    )
    state.stage = "reservation"
    files.deferred(state.reserve, state.paths.attempt, identity)
    state.stage = "input_retention"
    inputs = _inputs(state, cleanup)
    state.admissions.start_cell(
        state.cell, state.attempt, state.selected.descriptor_bytes, state.binding_bytes
    )
    state.completer = original._enter(
        cleanup,
        hold_series_cell(
            state.attempt, state.paths.public_summary, expected_identity=identity
        ),
    )
    return inputs


def _commands(state, inputs):
    return tuple(
        series_child_command(
            state.public,
            role,
            cell_ordinal=state.cell.ordinal,
            cell_binding_sha256=inputs.computational.binding_sha256,
        )
        for role in ("service", "client")
    )


async def _observe(state, commands):
    with held_attempt_writer(state.attempt, names=original._PARENT_NAMES) as writer:

        def retain(attempt, name, content):
            context.require(attempt is state.attempt)
            writer.retain(name, content)
            if name == "process-pair-intent.json":
                state.pair_intent_bytes = content

        try:
            state.observation = await observe_series_operational_children(
                state.attempt,
                service_command=commands[0],
                client_command=commands[1],
                deadlines=state.public.deadlines,
                writer=retain,
                series_admissions=state.admissions.issue,
            )
        except BaseException as error:
            state.original = error
            raise


def _complete(state, inputs, commands):
    state.stage = "completion"
    candidate = state.completer.complete(
        inputs=inputs,
        profile_bytes=state.public.profile_bytes,
        expected_profile_sha256=state.public.profile_sha256,
        expected_metadata_sha256=context.digest(state.metadata_bytes),
        internal_snapshot=state.internal_snapshot,
        external_snapshot=state.external_snapshot,
        observation=state.observation,
        service_command=commands[0],
        client_command=commands[1],
        expected_deadlines=state.public.deadlines,
    )
    state.stage = "finalization"
    return candidate


async def _run(state):
    with CleanupStack() as cleanup:
        try:
            context.validate(state)
            files.deferred(recheck_series_public_execution, state.public)
            cleanup.callback(
                files.deferred, recheck_series_public_execution, state.public
            )
            files.deferred(context.original.absent_outputs, state.paths)
            inputs = _prepare(state, cleanup)
            commands = _commands(state, inputs)
            state.stage = "observation"
            await _observe(state, commands)
            return _complete(state, inputs, commands)
        except BaseException as error:
            if state.original is None:
                state.original = error
            raise


async def _run_series_cell(
    public, metadata_bytes, internal_snapshot, external_snapshot, cell, *, admissions
):
    state = SeriesCellProgress(
        public, metadata_bytes, internal_snapshot, external_snapshot, cell, admissions
    )
    try:
        candidate = await _run(state)
        admissions.accept_cell(candidate, state.observation, state.pair_intent_bytes)
        return candidate
    except BaseException as error:
        reject(state, error)
