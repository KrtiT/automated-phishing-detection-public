"""Hold admitted series computations for the unchanged operational role bodies."""

import os
import sys
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path

from . import _operational_child_context as context
from . import _operational_input_schema as schema
from . import _study_preparation_files as files
from . import execution_receipt as receipt
from ._checkpoint_codec import canonical_bytes
from ._exception_cleanup import CleanupStack
from ._external_source_profile import _execution
from ._operational_child_io import held_writer, service_handles
from ._prepared_failure_context import carry_failure_context
from ._study_series_child_context import _descriptors, recheck_held_child, require
from ._study_series_child_transport import hold_series_child_inputs
from .operational_cell_inputs import RestoredOperationalCell
from .operational_role_records import OperationalRoleContext
from .study_series_inputs import SeriesOperationalCell


def _paths(held, arguments):
    frame = held.admission.frame
    require(arguments.role in ("service", "client") and frame.role == arguments.role)
    require(type(arguments.cell_ordinal) is int)
    require(arguments.cell_ordinal == frame.cell_ordinal)
    require(arguments.expected_binding_sha256 == frame.cell_binding_sha256)
    profile = schema.loads(held.authorization.profile_bytes)
    selected = profile["paths"]
    stem = f"cell-{arguments.cell_ordinal:03d}"
    paths = (
        Path(selected["cells_dir"]) / f"{stem}-attempt",
        Path(selected["historical_inputs_dir"]),
        Path(selected["cells_dir"]) / f"{stem}-inputs",
    )
    require(held.environment["APD_ATTEMPT_DIRECTORY"] == str(paths[0]))
    context._paths(held.authorization.base, *paths)
    return paths


def _validate(held, restored):
    require(type(restored) is SeriesOperationalCell)
    inputs, binding, frame = (
        restored.computational,
        held.authorization,
        held.admission.frame,
    )
    require(type(inputs) is RestoredOperationalCell)
    require(inputs.cell.ordinal == frame.cell_ordinal)
    require(inputs.binding_sha256 == frame.cell_binding_sha256)
    require(sha256(inputs.accepted_bytes).hexdigest() == frame.accepted_inputs_sha256)
    require(inputs.operational_profile_sha256 == binding.operational.profile_sha256)
    require(
        canonical_bytes(inputs.execution)
        == canonical_bytes(_execution(binding.base, dict(binding.base.source_hashes)))
    )
    payloads = dict(held.prefix_payloads)
    require("segment/history-import.json" in payloads)
    imported = schema.loads(payloads["segment/history-import.json"])
    require(
        sha256(restored.origin_metadata_bytes).hexdigest()
        == imported.get("origin_metadata_sha256")
    )
    return inputs


def _resources(cleanup, held, arguments, paths, actual):
    environment = held.environment
    handles = ()
    if arguments.role == "service":
        manager = service_handles(_descriptors(environment, "service"), actual.base_url)
        cleanup.push(manager.__exit__)
        handles = manager.__enter__()
    attempt = receipt.Attempt(paths[0], environment["APD_RESERVATION_SHA256"])
    manager = held_writer(attempt, arguments.role)
    cleanup.push(manager.__exit__)
    writer = manager.__enter__()
    return handles, writer


def _acquire(cleanup, held, arguments):
    paths = _paths(held, arguments)
    actual = OperationalRoleContext(
        os.getpid(), (sys.executable, *sys.argv), held.environment["APD_BASE_URL"]
    )
    handles, writer = _resources(cleanup, held, arguments, paths, actual)
    manager = hold_series_child_inputs(
        paths[1],
        paths[2],
        profile_bytes=held.authorization.profile_bytes,
        frame=held.admission.frame,
        expected_cell_reservation_sha256=held.environment["APD_RESERVATION_SHA256"],
    )
    cleanup.push(manager.__exit__)
    inputs = _validate(held, manager.__enter__())
    files.deferred(recheck_held_child, held)
    return context.HeldChild(inputs, actual, writer.retain, handles)


@contextmanager
def held_operational(held, arguments):
    original = None
    try:
        with CleanupStack() as cleanup:
            files.deferred(recheck_held_child, held)
            cleanup.callback(files.deferred, recheck_held_child, held)
            try:
                runtime = _acquire(cleanup, held, arguments)
                yield runtime
            except BaseException as error:
                original = error
                raise
    except BaseException as error:
        carry_failure_context(error, original)
        raise
