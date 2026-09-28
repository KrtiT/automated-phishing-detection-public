"""Study-only cell wiring reuses original held inputs, processes and acceptance."""

from hashlib import sha256

from . import operational_cell_runner as original
from ._study_child_commands import cell_command, require
from ._study_run_paths import cell_paths
from .study_execution import recheck_study_execution


def commands_and_admissions(admissions, inputs):
    commands = tuple(
        cell_command(
            admissions.authorization, role, inputs.cell.ordinal, inputs.binding_sha256
        )
        for role in ("service", "client")
    )

    def issue(role, command):
        require(role in ("service", "client"))
        require(command == commands[0 if role == "service" else 1])
        return admissions.issue(
            role,
            command,
            predecessor_sha256=admissions.source_results_sha256,
            accepted_inputs_sha256=sha256(inputs.accepted_bytes).hexdigest(),
            cell_binding_sha256=inputs.binding_sha256,
        )

    return commands, issue


def _validate(authorization, cell, paths, admissions):
    from ._adopted_study_ledger import AdmissionLedger
    from .study_execution import StudyExecutionBinding

    require(type(authorization) is StudyExecutionBinding)
    require(type(admissions) is AdmissionLedger)
    require(admissions.authorization is authorization)
    require(paths == cell_paths(authorization.paths, cell.ordinal))


async def run_adopted_cell(authorization, accepted, cell, *, paths, admissions):
    _validate(authorization, cell, paths, admissions)
    state = original.CellProgress()
    try:
        recheck_study_execution(authorization)
        result = await original._run(
            state,
            authorization.base,
            authorization.operational,
            accepted,
            cell,
            paths,
            authorization.paths.internal.artifacts,
            authorization.deadlines,
            study_admissions=admissions,
        )
        recheck_study_execution(authorization)
        return result
    except BaseException as error:
        original.reject(state, error)
