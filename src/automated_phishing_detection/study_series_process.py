"""Observe admitted series children, never grant access or accept research cells."""

import asyncio

from . import operational_process as original
from ._study_series_process_children import SeriesOwnedChildren


async def _observe(observations, deadlines, service_command, client_command, issuer):
    failure, children = None, None
    try:
        with SeriesOwnedChildren(observations, deadlines, issuer) as children:
            failure = await original._attempt(
                children, observations, service_command, client_command
            )
            cancelled, progress = await original._finish_cleanup(children, observations)
            failure = original._admission_failure(children, failure)
            return original._result(observations, failure, cancelled, progress)
    except BaseException as error:
        failure = original._admission_failure(children, failure)
        observations.fail("process_cleanup_failed")
        if isinstance(failure, asyncio.CancelledError):
            raise original.OperationalProcessCancelled(
                progress=observations.snapshot()
            ) from failure
        selected = (
            failure if isinstance(failure, (KeyboardInterrupt, SystemExit)) else error
        )
        selected.progress = observations.snapshot()
        raise selected from None


def _validate(attempt, commands, deadlines, writer, issuer):
    if type(deadlines) is not dict or set(deadlines) != {
        "startup",
        "shutdown",
        "terminate",
        "kill",
    }:
        raise original.OperationalProcessError("invalid_protective_deadline")
    original._validate(attempt, commands, deadlines)
    if not callable(writer):
        raise original.OperationalProcessError("invalid_process_writer")
    if not callable(issuer):
        raise original.OperationalProcessError("invalid_series_admissions")


async def observe_series_operational_children(
    attempt, *, service_command, client_command, deadlines, writer, series_admissions
):
    _validate(
        attempt, (service_command, client_command), deadlines, writer, series_admissions
    )
    observations = original.Observations(attempt, writer)
    try:
        observations.claim(service_command, client_command, deadlines)
    except BaseException as error:
        original._retain_claim_failure(observations, error)
        raise
    return await _observe(
        observations, deadlines, service_command, client_command, series_admissions
    )
