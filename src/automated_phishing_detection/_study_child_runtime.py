"""Dispatch only fixed roles after live parent and complete-profile admission."""

from ._study_child_context import held_authorization, require
from ._study_child_sources import run_external, run_internal


async def run_operational(held, arguments):
    from ._study_child_operational import run_operational as run

    return await run(held, arguments)


async def run_child(arguments):
    with held_authorization(arguments) as held:
        if arguments.role == "internal":
            return run_internal(held)
        if arguments.role == "external":
            return run_external(held, arguments)
        require(arguments.role in ("service", "client"))
        return await run_operational(held, arguments)
