"""Live series admission and committed adoption precede child private inputs."""

import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial

from . import _operational_input_schema as schema
from . import _study_preparation_files as files
from ._exception_cleanup import CleanupStack
from ._prepared_failure_context import carry_failure_context
from ._process_support import _defer_interrupt
from ._study_authorized_cli import keywords
from ._study_series_admission import consume_series_admission
from ._study_series_child_commands import series_child_command
from ._study_series_child_prefix import hold_series_child_prefix
from .study_series_execution import (
    bind_series_public_execution,
    recheck_series_public_execution,
)

LOCATOR = "APD_STUDY_SERIES_ADMISSION_FD"
COMMON = {"APD_BASE_URL", "APD_ATTEMPT_DIRECTORY", "APD_RESERVATION_SHA256"}
HANDLES = ("APD_LISTENER_FD", "APD_STOP_FD", "APD_READY_FD")


def require(condition):
    if not condition:
        raise ValueError("invalid_series_child_context")


def _environment(role):
    require(role in ("service", "client"))
    expected = {LOCATOR, *COMMON}
    if role == "service":
        expected.update(HANDLES)
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
    descriptors = tuple(map(int, values))
    require(len(set(descriptors)) == 3 and min(descriptors) >= 3)
    return descriptors


def _require_execution_policy(binding):
    if (
        schema.loads(binding.policy_bytes)["status"]
        != "specified_for_explicit_adoption"
    ):
        raise ValueError("series_child_execution_not_adopted")


def _recheck(binding, child):
    recheck_series_public_execution(binding)
    _require_execution_policy(binding)
    child.check()


@dataclass(frozen=True)
class HeldSeriesChild:
    authorization: object
    admission: object
    environment: dict
    prefix_payloads: tuple


def recheck_held_child(held):
    _recheck(held.authorization, held.admission)


def _admission(cleanup, arguments, environment):
    with _defer_interrupt():
        child = consume_series_admission(
            arguments.role,
            (sys.executable, *sys.argv),
            environment=environment,
            separated_fds=_descriptors(environment, arguments.role),
        )
        cleanup.callback(files.deferred, child.close)
    cleanup.callback(files.deferred, child.check)
    return child


def _join(arguments, binding, child):
    frame = child.frame
    require(frame.profile_sha256 == binding.profile_sha256)
    require(frame.envelope_sha256 == binding.envelope_sha256)
    require(frame.cell_ordinal == arguments.cell_ordinal)
    require(frame.cell_binding_sha256 == arguments.expected_binding_sha256)
    command = series_child_command(
        binding,
        arguments.role,
        cell_ordinal=arguments.cell_ordinal,
        cell_binding_sha256=arguments.expected_binding_sha256,
    )
    require(command == (sys.executable, *sys.argv))


def _acquire(cleanup, arguments):
    environment = _environment(arguments.role)
    child = _admission(cleanup, arguments, environment)
    binding = files.deferred(
        partial(
            bind_series_public_execution,
            arguments.repo_root,
            expected_profile_sha256=arguments.expected_profile_sha256,
            **keywords(arguments),
        )
    )
    cleanup.callback(files.deferred, recheck_series_public_execution, binding)
    _require_execution_policy(binding)
    _join(arguments, binding, child)
    files.deferred(_recheck, binding, child)
    manager = hold_series_child_prefix(binding, child.frame)
    cleanup.push(manager.__exit__)
    payloads = manager.__enter__()
    files.deferred(child.check)
    return HeldSeriesChild(binding, child, environment, payloads)


@contextmanager
def held_authorization(arguments):
    original = None
    try:
        with CleanupStack() as cleanup:
            try:
                held = _acquire(cleanup, arguments)
                yield held
            except BaseException as error:
                original = error
                raise
    except BaseException as error:
        carry_failure_context(error, original)
        raise
