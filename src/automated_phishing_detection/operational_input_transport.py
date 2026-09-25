"""Held immutable operational inputs; matching bytes confer no execution authority."""

from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path

from . import _operational_input_files as storage
from . import _study_preparation_files as files
from . import execution_receipt as receipt
from ._external_records_validation import digest
from ._prepared_failure_context import carry_failure_context


class OperationalInputTransportError(ValueError):
    """Symbolic rejection without private content, paths or parser diagnostics."""


def _reject(error, original, *, yielded=True):
    carry_failure_context(error, original)
    if not isinstance(error, Exception) or (yielded and error is original):
        raise error from None
    rejected = OperationalInputTransportError("invalid_operational_input_transport")
    carry_failure_context(rejected, error)
    raise rejected from None


@contextmanager
def _retain(directory, payloads):
    original = None
    try:
        files.require(all(type(content) is bytes for _, content in payloads))
        path = receipt._absolute_path(directory)
        with storage.retain(path, payloads) as held:
            try:
                yield held
            except BaseException as error:
                original = error
                raise
    except BaseException as error:
        _reject(error, original)


@contextmanager
def retain_operational_root_inputs(directory: Path, *, accepted_inputs: bytes):
    """Create the one-file private input leaf; preserve every partial artifact."""
    with _retain(directory, (("accepted-inputs.json", accepted_inputs),)) as held:
        yield held


@contextmanager
def retain_operational_cell_inputs(
    directory: Path, *, descriptor: bytes, binding: bytes, manifest: bytes
):
    """Create three immutable cell inputs without constraining mutable attempts."""
    payloads = (
        ("descriptor.json", descriptor),
        ("binding.json", binding),
        ("manifest", manifest),
    )
    with _retain(directory, payloads) as held:
        yield held


def _paths(accepted, cell):
    root_path, cell_path = (
        receipt._absolute_path(accepted),
        receipt._absolute_path(cell),
    )
    files.require(root_path != cell_path)
    files.require(
        root_path not in cell_path.parents and cell_path not in root_path.parents
    )
    return root_path, cell_path


def _contents(held, expected):
    root, cell = held
    directory, states = cell
    binding = files.deferred(
        storage.read_file, directory, "binding.json", states["binding.json"]
    )
    files.require(sha256(binding).hexdigest() == expected)
    contents = {"binding.json": binding}
    for (directory, states), name in (
        (cell, "descriptor.json"),
        (root, "accepted-inputs.json"),
        (cell, "manifest"),
    ):
        contents[name] = files.deferred(
            storage.read_file, directory, name, states[name]
        )
    files.deferred(storage.check_all, held)
    return contents


def _restore(contents, expectations):
    from .operational_cell_inputs import restore_cell_inputs

    return restore_cell_inputs(
        contents["accepted-inputs.json"],
        contents["descriptor.json"],
        contents["binding.json"],
        contents["manifest"],
        **expectations,
    )


def _restore_held(held, expected_binding, expected_reservation):
    contents = _contents(held, expected_binding)
    restored = _restore(
        contents,
        {
            "expected_binding_sha256": expected_binding,
            "expected_cell_reservation_sha256": expected_reservation,
        },
    )
    files.deferred(storage.check_all, held)
    return restored


@contextmanager
def hold_operational_inputs(
    accepted_inputs_directory: Path,
    cell_input_directory: Path,
    *,
    expected_binding_sha256: str,
    expected_cell_reservation_sha256: str,
):
    """Yield pure restored context while all original input file identities remain held."""
    original, yielded = None, False
    try:
        digest(expected_binding_sha256)
        digest(expected_cell_reservation_sha256)
        paths = _paths(accepted_inputs_directory, cell_input_directory)
        with storage.hold(*paths) as held:
            try:
                restored = _restore_held(
                    held, expected_binding_sha256, expected_cell_reservation_sha256
                )
                yielded = True
                yield restored
            except BaseException as error:
                original = error
                raise
    except BaseException as error:
        _reject(error, original, yielded=yielded)
