"""Use the original operational owner/replay under a live adopted-root context."""

from contextlib import contextmanager
from functools import partial
from hashlib import sha256

from . import _operational_child_context as context
from . import operational_cell_client as client
from . import operational_cell_service as service
from ._study_child_context import LOCATOR, recheck_held_child, require
from ._study_run_paths import cell_paths


def _context(held, arguments):
    authorization, frame = held.authorization, held.admission.frame
    require(arguments.role in ("service", "client"))
    require(arguments.expected_binding_sha256 == frame.cell_binding_sha256)
    selected = cell_paths(authorization.paths, arguments.cell_ordinal)
    environment = {
        name: value for name, value in held.environment.items() if name != LOCATOR
    }
    require(environment["APD_ATTEMPT_DIRECTORY"] == str(selected.attempt))
    paths = (
        selected.attempt,
        selected.accepted_inputs_directory,
        selected.cell_input_directory,
    )
    context._paths(authorization.base, *paths)
    return context._held(
        authorization.base,
        authorization.operational,
        arguments.role,
        environment,
        paths,
        arguments.expected_binding_sha256,
        lifecycle_check=partial(recheck_held_child, held),
    )


@contextmanager
def held_operational(held, arguments):
    with _context(held, arguments) as runtime:
        require(runtime.inputs.cell.ordinal == arguments.cell_ordinal)
        require(
            sha256(runtime.inputs.accepted_bytes).hexdigest()
            == held.admission.frame.accepted_inputs_sha256
        )
        held.admission.check()
        yield runtime


async def _client(runtime, lifecycle_check):
    try:
        lifecycle_check()
        content = await client._replay(runtime)
        lifecycle_check()
        runtime.retain("run.json", content)
    except BaseException as error:
        client._failed(runtime, error)


async def run_operational(held, arguments):
    with held_operational(held, arguments) as runtime:
        lifecycle_check = partial(recheck_held_child, held)
        if arguments.role == "client":
            return await _client(runtime, lifecycle_check)
        return await service._serve(
            runtime,
            held.authorization.base,
            held.authorization.paths.internal.artifacts,
            lifecycle_check=lifecycle_check,
        )
