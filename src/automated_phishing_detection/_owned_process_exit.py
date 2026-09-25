"""Observe an owned child status without trusting Popen's cached return code."""

import os
from dataclasses import dataclass
from subprocess import Popen


class OwnedProcessExitError(ValueError):
    """An invalid process identity or wait result cannot establish an exit."""


@dataclass(frozen=True)
class OwnedProcessExit:
    pid: int
    exit_observed: bool
    exit_code: int | None


def _observation(result: object, pid: int, block: bool) -> OwnedProcessExit | None:
    if type(result) is not tuple or len(result) != 2:
        raise OwnedProcessExitError("invalid_wait_result")
    waited_pid, status = result
    if type(waited_pid) is not int or type(status) is not int or status < 0:
        raise OwnedProcessExitError("invalid_wait_result")
    if waited_pid == 0 and status == 0 and not block:
        return None
    if waited_pid != pid:
        raise OwnedProcessExitError("invalid_wait_result")
    try:
        code = os.waitstatus_to_exitcode(status)
    except (ValueError, OverflowError):
        raise OwnedProcessExitError("invalid_wait_result") from None
    return OwnedProcessExit(pid, True, code)


def observe_owned_exit(
    process: Popen, *, block: bool = False
) -> OwnedProcessExit | None:
    """Return pending, observed exit, or terminal loss of child ownership."""
    if type(block) is not bool:
        raise OwnedProcessExitError("invalid_wait_mode")
    pid = process.pid
    if type(pid) is not int or pid <= 0:
        raise OwnedProcessExitError("invalid_process_pid")
    try:
        result = os.waitpid(pid, 0 if block else os.WNOHANG)
    except ChildProcessError:
        return OwnedProcessExit(pid, False, None)
    observed = _observation(result, pid, block)
    if observed is not None:
        process.returncode = observed.exit_code
    return observed
