"""Private already-bound cell composition; no public independent-cell admission.

Only the observing parent can accept the returned pair after every holder exits.
Explicit fixture deadlines do not adopt the closed candidate profile.
"""

from dataclasses import dataclass, field
from hashlib import sha256

from . import _operational_cell_runner_context as context
from . import _study_preparation_files as files
from ._exception_cleanup import CleanupStack
from ._operational_attempt_io import held_attempt_writer
from ._operational_cell_commands import build_cell_commands
from ._operational_cell_failure import CellProgress, OperationalCellFailure, reject
from ._operational_cell_protocol import WORKING_NAMES
from ._operational_cell_results import VerifiedOperationalCell
from ._operational_cell_runner_context import (
    OperationalCellExecutionError,
    OperationalCellPaths,
)
from ._operational_process_records import ProcessObservation
from .execution_preflight import recheck_binding
from .operational_cell_acceptance import cell_identity
from .operational_cell_completion import hold_operational_cell
from .operational_cell_inputs import bind_cell_descriptor, build_cell_descriptor
from .operational_input_transport import (
    hold_operational_inputs,
    retain_operational_cell_inputs,
)
from .operational_process import _observe_pair_with_writer

__all__ = [
    "OperationalCellPaths",
    "OperationalCellExecutionError",
    "OperationalCellFailure",
    "ObservedOperationalCell",
]

_CHILD_NAMES = {
    "service-role.json",
    "client-role.json",
    "service-ready.json",
    "service-cleanup.json",
    "warmup.json",
    "measured.json",
    "run.json",
}
_PARENT_NAMES = tuple(
    name for name in WORKING_NAMES if name not in _CHILD_NAMES | {"reservation.json"}
)


@dataclass(frozen=True)
class ObservedOperationalCell:
    observation: ProcessObservation = field(repr=False)
    snapshot: VerifiedOperationalCell = field(repr=False)


def _enter(cleanup, manager):
    cleanup.push(manager.__exit__)
    return manager.__enter__()


def _inputs(cleanup, paths, accepted, selected, attempt):
    binding_bytes = bind_cell_descriptor(
        selected.descriptor_bytes, cell_reservation_sha256=attempt.reservation_sha256
    )
    _enter(
        cleanup,
        retain_operational_cell_inputs(
            paths.cell_input_directory,
            descriptor=selected.descriptor_bytes,
            binding=binding_bytes,
            manifest=selected.manifest_bytes,
        ),
    )
    inputs = _enter(
        cleanup,
        hold_operational_inputs(
            paths.accepted_inputs_directory,
            paths.cell_input_directory,
            expected_binding_sha256=sha256(binding_bytes).hexdigest(),
            expected_cell_reservation_sha256=attempt.reservation_sha256,
        ),
    )
    context.same_inputs(inputs, accepted, selected, binding_bytes)
    return inputs


async def _observe(state, commands, deadlines):
    with held_attempt_writer(state.attempt, names=_PARENT_NAMES) as writer:

        def retain(attempt, name, content):
            context.require(attempt is state.attempt)
            writer.retain(name, content)

        try:
            state.observation = await _observe_pair_with_writer(
                state.attempt,
                service_command=commands[0],
                client_command=commands[1],
                deadlines=deadlines,
                writer=retain,
            )
        except BaseException as error:
            state.original = error
            raise


async def _execute(
    state, cleanup, binding, profile, accepted, selected, paths, artifacts, deadlines
):
    identity = cell_identity(accepted, selected.descriptor_bytes)
    state.stage = "reservation"
    files.deferred(state.reserve, paths.attempt, identity)
    state.stage = "input_retention"
    inputs = _inputs(cleanup, paths, accepted, selected, state.attempt)
    state.completer = _enter(
        cleanup,
        hold_operational_cell(
            state.attempt,
            paths.public_summary,
            expected_identity=identity,
        ),
    )
    commands = build_cell_commands(
        binding,
        profile,
        accepted_inputs_directory=paths.accepted_inputs_directory,
        cell_input_directory=paths.cell_input_directory,
        expected_binding_sha256=inputs.binding_sha256,
        artifacts=artifacts,
    )
    state.stage = "observation"
    await _observe(state, commands, deadlines)
    return _complete(state, inputs, accepted, commands, deadlines)


def _complete(state, inputs, accepted, commands, deadlines):
    state.stage = "completion"
    snapshot = state.completer.complete(
        inputs=inputs,
        accepted=accepted,
        observation=state.observation,
        service_command=commands[0],
        client_command=commands[1],
        expected_deadlines=deadlines,
    )
    state.stage = "finalization"
    return ObservedOperationalCell(state.observation, snapshot)


async def _run(state, binding, profile, accepted, cell, paths, artifacts, deadlines):
    with CleanupStack() as cleanup:
        try:
            context.validate(
                binding, profile, accepted, cell, paths, artifacts, deadlines
            )
            state.cell = cell
            files.deferred(recheck_binding, binding)
            cleanup.callback(files.deferred, recheck_binding, binding)
            files.deferred(context.absent_outputs, paths)
            state.stage = "selection"
            selected = build_cell_descriptor(accepted, cell)
            return await _execute(
                state,
                cleanup,
                binding,
                profile,
                accepted,
                selected,
                paths,
                artifacts,
                deadlines.copy(),
            )
        except BaseException as error:
            if state.original is None:
                state.original = error
            raise


async def _run_bound_cell(
    binding, profile, accepted, cell, *, paths, artifacts, deadlines
):
    state = CellProgress()
    try:
        return await _run(
            state, binding, profile, accepted, cell, paths, artifacts, deadlines
        )
    except BaseException as error:
        reject(state, error)
