"""Run unchanged operational bodies only inside a held series child context."""

from dataclasses import fields
from functools import partial
from pathlib import Path

from . import _operational_input_schema as schema
from . import operational_cell_client as client
from . import operational_cell_service as service
from ._study_series_child_context import (
    held_authorization,
    recheck_held_child,
    require,
)
from ._study_series_child_operational import held_operational
from .bound_models import ArtifactPaths


def _artifacts(binding):
    profile = schema.loads(binding.profile_bytes)
    paths = profile["origin"]["profile"]["paths"]
    return ArtifactPaths(
        **{
            selected.name: Path(paths[selected.name.replace("_", "-")])
            for selected in fields(ArtifactPaths)
        }
    )


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
            _artifacts(held.authorization),
            lifecycle_check=lifecycle_check,
        )


async def run_child(arguments):
    require(arguments.role in ("service", "client"))
    with held_authorization(arguments) as held:
        return await run_operational(held, arguments)
