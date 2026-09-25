"""Closed child admission and held context; private bodies grant no authority."""

import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from . import _study_preparation_files as files
from . import execution_receipt as receipt
from ._checkpoint_codec import canonical_bytes
from ._exception_cleanup import CleanupStack
from ._external_source_profile import _execution
from ._operational_child_io import (
    OperationalChildError,
    held_writer,
    require,
    service_handles,
)
from ._prepared_failure_context import carry_failure_context
from .execution_preflight import bind_execution, recheck_binding
from .operational_input_transport import hold_operational_inputs
from .operational_role_records import OperationalRoleContext


def resolve_operational_profile(binding):
    from ._operational_profile import resolve_operational_profile as resolve

    return resolve(binding)


def public_context(root, revision, contract, expected_profile):
    binding = files.deferred(
        partial(
            bind_execution,
            root,
            expected_revision=revision,
            expected_contract_sha256=contract,
        )
    )
    profile = files.deferred(resolve_operational_profile, binding)
    require(
        type(expected_profile) is str
        and re.fullmatch(r"[0-9a-f]{64}", expected_profile)
    )
    require(profile.profile_sha256 == expected_profile)
    if not binding.protected_evaluation_ready or not profile.protected_evaluation_ready:
        raise OperationalChildError("pre_access_freeze_incomplete")
    return binding, profile


def _environment(role):
    common = {"APD_BASE_URL", "APD_ATTEMPT_DIRECTORY", "APD_RESERVATION_SHA256"}
    names = common | (
        {"APD_LISTENER_FD", "APD_STOP_FD", "APD_READY_FD"}
        if role == "service"
        else set()
    )
    actual = {
        name: value for name, value in os.environ.items() if name.startswith("APD_")
    }
    require(set(actual) == names)
    return actual


def _paths(binding, attempt, accepted, cell):
    paths = (attempt, accepted, cell)
    require(
        all(
            isinstance(path, Path) and path.is_absolute() and ".." not in path.parts
            for path in paths
        )
    )
    for path in paths:
        require(path != binding.root and binding.root not in path.parents)
    require(len(set(paths)) == 3)
    require(
        all(
            first not in second.parents
            for first in paths
            for second in paths
            if first != second
        )
    )


def _descriptors(environment, role):
    if role != "service":
        return ()
    values = tuple(
        environment[name] for name in ("APD_LISTENER_FD", "APD_STOP_FD", "APD_READY_FD")
    )
    require(all(re.fullmatch(r"0|[1-9][0-9]*", value) for value in values))
    descriptors = tuple(map(int, values))
    require(len(set(descriptors)) == 3 and min(descriptors) >= 3)
    return descriptors


@dataclass(frozen=True)
class HeldChild:
    inputs: object
    role_context: OperationalRoleContext
    retain: object
    handles: tuple


def _acquire(cleanup, binding, profile, role, environment, paths, digest):
    attempt_path, accepted, cell = paths
    actual = OperationalRoleContext(
        os.getpid(), (sys.executable, *sys.argv), environment["APD_BASE_URL"]
    )
    handles = ()
    if role == "service":
        manager = service_handles(_descriptors(environment, role), actual.base_url)
        cleanup.push(manager.__exit__)
        handles = manager.__enter__()
    attempt = receipt.Attempt(attempt_path, environment["APD_RESERVATION_SHA256"])
    writer_context = held_writer(attempt, role)
    cleanup.push(writer_context.__exit__)
    writer = writer_context.__enter__()
    context = hold_operational_inputs(
        accepted,
        cell,
        expected_binding_sha256=digest,
        expected_cell_reservation_sha256=attempt.reservation_sha256,
    )
    cleanup.push(context.__exit__)
    inputs = context.__enter__()
    require(inputs.operational_profile_sha256 == profile.profile_sha256)
    require(
        canonical_bytes(inputs.execution)
        == canonical_bytes(_execution(binding, dict(binding.source_hashes)))
    )
    return HeldChild(inputs, actual, writer.retain, handles)


def held_child(
    binding,
    profile,
    role,
    *,
    accepted_inputs_directory,
    cell_input_directory,
    expected_binding_sha256,
):
    environment = _environment(role)
    paths = (
        Path(environment["APD_ATTEMPT_DIRECTORY"]),
        accepted_inputs_directory,
        cell_input_directory,
    )
    _paths(binding, *paths)
    return _held(binding, profile, role, environment, paths, expected_binding_sha256)


@contextmanager
def _held(binding, profile, role, environment, paths, expected_binding_sha256):
    original = None
    try:
        with CleanupStack() as cleanup:
            files.deferred(recheck_binding, binding)
            cleanup.callback(files.deferred, recheck_binding, binding)
            try:
                held = _acquire(
                    cleanup,
                    binding,
                    profile,
                    role,
                    environment,
                    paths,
                    expected_binding_sha256,
                )
                yield held
            except BaseException as error:
                original = error
                raise
    except BaseException as error:
        carry_failure_context(error, original)
        raise
