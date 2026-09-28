"""A live parent channel and exact approved profile precede all child inputs."""

import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial

from . import _study_preparation_files as files
from ._exception_cleanup import CleanupStack
from ._prepared_failure_context import carry_failure_context
from ._process_support import _defer_interrupt
from ._study_admission import consume_child_admission
from ._study_authorized_cli import keywords
from ._study_child_root import hold_child_root
from .study_execution import bind_study_execution, recheck_study_execution

LOCATOR = "APD_STUDY_ADMISSION_FD"
COMMON = {"APD_BASE_URL", "APD_ATTEMPT_DIRECTORY", "APD_RESERVATION_SHA256"}
HANDLES = ("APD_LISTENER_FD", "APD_STOP_FD", "APD_READY_FD")


def require(condition):
    if not condition:
        raise ValueError("invalid_study_child_context")


def _environment(role):
    require(role in ("internal", "external", "service", "client"))
    expected = {LOCATOR}
    if role in ("service", "client"):
        expected |= COMMON
    if role == "service":
        expected |= set(HANDLES)
    actual = {
        name: value for name, value in os.environ.items() if name.startswith("APD_")
    }
    require(set(actual) == expected)
    return actual


def _descriptors(environment, role):
    if role != "service":
        return ()
    values = tuple(environment[name] for name in HANDLES)
    require(
        all(
            type(value) is str and re.fullmatch(r"[1-9][0-9]*", value)
            for value in values
        )
    )
    result = tuple(map(int, values))
    require(len(set(result)) == 3 and min(result) >= 3)
    return result


@dataclass(frozen=True)
class HeldStudyChild:
    authorization: object
    admission: object
    environment: dict
    root_payloads: dict


def recheck_held_child(held):
    recheck_study_execution(held.authorization)
    held.admission.check()


def _admission(cleanup, arguments, environment):
    with _defer_interrupt():
        child = consume_child_admission(
            arguments.role,
            (sys.executable, *sys.argv),
            environment=environment,
            separated_fds=_descriptors(environment, arguments.role),
        )
        cleanup.callback(files.deferred, child.close)
    cleanup.callback(files.deferred, child.check)
    return child


def _acquire(cleanup, arguments):
    environment = _environment(arguments.role)
    child = _admission(cleanup, arguments, environment)
    authorization = files.deferred(
        partial(bind_study_execution, arguments.repo_root, **keywords(arguments))
    )
    require(child.frame.profile_sha256 == authorization.profile_sha256)
    require(child.frame.envelope_sha256 == authorization.envelope_sha256)
    cleanup.callback(files.deferred, recheck_study_execution, authorization)
    manager = hold_child_root(authorization, child.frame)
    cleanup.push(manager.__exit__)
    payloads = manager.__enter__()
    files.deferred(child.check)
    return HeldStudyChild(authorization, child, environment, payloads)


@contextmanager
def held_authorization(arguments):
    original = None
    try:
        with CleanupStack() as cleanup:
            held = _acquire(cleanup, arguments)
            try:
                yield held
            except BaseException as error:
                original = error
                raise
    except BaseException as error:
        carry_failure_context(error, original)
        raise
