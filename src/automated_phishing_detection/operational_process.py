"""Observe one fresh subprocess pair; never authenticate or accept research cells."""

import asyncio
import math
import os

from . import execution_receipt as receipt
from ._operational_process_children import OwnedChildren, _defer_interrupt
from ._operational_process_records import (
    Observations,
    OperationalProcessError,
    ProcessObservation,
    _record,
)


class OperationalProcessCancelled(asyncio.CancelledError):
    def __init__(self, *, progress):
        self.progress = progress
        super().__init__("parent_cancelled")


def process_progress(error):
    """Recover immutable observations even through Python 3.10 cancellation."""
    seen = set()
    while isinstance(error, BaseException) and id(error) not in seen:
        seen.add(id(error))
        progress = getattr(error, "progress", None)
        if type(progress) is bytes:
            return progress
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return None


def _validate(attempt, commands, deadlines):
    if type(attempt) is not receipt.Attempt:
        raise OperationalProcessError("invalid_attempt")
    for command in commands:
        if (
            type(command) is not tuple
            or not command
            or any(
                type(argument) is not str or not argument or "\0" in argument
                for argument in command
            )
        ):
            raise OperationalProcessError("invalid_command")
    for value in deadlines.values():
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise OperationalProcessError("invalid_protective_deadline")


async def _ready(children, observations):
    received = bytearray()
    while True:
        if children.exited("service"):
            raise OperationalProcessError("service_startup_exit")
        try:
            chunk = os.read(children.ready_read, 4096)
        except BlockingIOError:
            chunk = None
        if chunk is not None:
            received.extend(chunk)
            if received == b"ready\n":
                observations.ready = observations.lifecycle(
                    "service-ready.json",
                    pid=children.processes["service"].pid,
                    port=children.port,
                )
                return
            if not chunk or len(received) >= 6:
                raise OperationalProcessError("invalid_ready_control")
        await asyncio.sleep(0.01)


async def _run(children, observations, service_command, client_command):
    children.launch("service", service_command)
    try:
        await asyncio.wait_for(
            _ready(children, observations), children.deadlines["startup"]
        )
    except asyncio.TimeoutError:
        raise OperationalProcessError("startup_timeout") from None
    if children.exited("service"):
        raise OperationalProcessError("service_startup_exit")
    children.launch("client", client_command)
    while not children.exited("client"):
        if children.exited("service"):
            raise OperationalProcessError("service_exit_unsuccessful")
        await asyncio.sleep(0.01)
    if observations.value["client"]["exit_code"] != 0:
        raise OperationalProcessError("client_exit_unsuccessful")


async def _cleanup(children, observations):
    task = asyncio.create_task(children.cleanup())
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
            observations.fail("parent_cancelled")
        except Exception:
            break
    try:
        task.result()
    except Exception:
        observations.fail("cleanup_failed")
    return cancelled


async def _finish_cleanup(children, observations):
    try:
        with _defer_interrupt() as interrupted:
            cancelled = await _cleanup(children, observations)
            if interrupted:
                observations.fail("parent_interrupted")
            progress = observations.finish(children)
    except KeyboardInterrupt as error:
        error.progress = observations.snapshot()
        raise
    return cancelled, progress


async def _observe(observations, deadlines, service_command, client_command):
    failure = None
    with OwnedChildren(observations, deadlines) as children:
        try:
            await _run(children, observations, service_command, client_command)
        except BaseException as error:
            failure = error
            observations.fail(
                "parent_cancelled"
                if isinstance(error, asyncio.CancelledError)
                else error.check_id
                if isinstance(error, OperationalProcessError)
                else "parent_interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "process_pair_failed"
            )
        cancelled, progress = await _finish_cleanup(children, observations)
    if cancelled or isinstance(failure, asyncio.CancelledError):
        raise OperationalProcessCancelled(progress=progress) from failure
    if isinstance(failure, (KeyboardInterrupt, SystemExit)):
        failure.progress = progress
        raise failure
    if observations.value["failure"] is not None:
        raise OperationalProcessError(
            observations.value["failure"], progress=progress
        ) from None
    return ProcessObservation(progress)


async def observe_process_pair(
    attempt,
    *,
    service_command,
    client_command,
    startup_timeout_seconds,
    shutdown_timeout_seconds,
    terminate_timeout_seconds,
    kill_timeout_seconds,
):
    """Consume one attempt; observe children, not source identity or scientific validity.

    The service installs service-ready.json before writing ready\\n to APD_READY_FD.
    Its existing lifecycle helper receives APD_LISTENER_FD and APD_STOP_FD.
    Both children receive APD_BASE_URL and the reserved attempt directory/hash.
    Protective deadlines must be explicitly frozen by the eventual bound runner.
    """
    deadlines = {
        "startup": startup_timeout_seconds,
        "shutdown": shutdown_timeout_seconds,
        "terminate": terminate_timeout_seconds,
        "kill": kill_timeout_seconds,
    }
    _validate(attempt, (service_command, client_command), deadlines)
    observations = Observations(attempt, _record)
    observations.claim(service_command, client_command, deadlines)
    return await _observe(observations, deadlines, service_command, client_command)
