"""Once-read preparation transport retained through observation and final checks.

Expected bytes and hashes preserve same-parent consistency, not execution
authority. No source acquisition, scoring, cleanup, retry or historical resume
is performed or authorized by this reader.
"""

import os
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path

from . import _study_preparation_files as files
from . import execution_receipt as receipt
from ._exception_cleanup import CleanupStack, preserve_cleanup
from ._external_records_validation import digest
from ._study_preparation_records import PreparedStudySnapshot
from .study_preparation_retention import PREPARATION_ORDER

_NAMES = ("reservation.json", *PREPARATION_ORDER)


class StudyPreparationTransportError(ValueError):
    """Symbolic rejection without private paths, content or parser diagnostics."""


def _read_file(directory, name, initial):
    with CleanupStack() as cleanup:
        descriptor = files.open_file(
            cleanup, directory, name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        )
        files.require(files.state(os.fstat(descriptor)) == initial)
        stream = cleanup.enter_context(os.fdopen(descriptor, "rb", closefd=False))
        content = stream.read()
        files.require(len(content) == initial[2])
        files.require(files.state(os.fstat(descriptor)) == initial)
        files.require(files.capture(directory, name) == initial)
        files.deferred(directory.check)
        return content


def _reservation(directory, content, identity, expected):
    files.require(sha256(content).hexdigest() == expected)
    files.require(
        content
        == receipt._json_bytes(
            {
                "schema_version": 1,
                "status": "reserved",
                "directory": str(directory.path),
                "identity": identity,
            },
            "preparation_reservation",
        )
    )


@contextmanager
def _hold_snapshot(path, identity, expected):
    with CleanupStack() as cleanup:
        directory = files.enter_directory(cleanup, path)
        states = {name: files.capture(directory, name) for name in _NAMES}
        files.check(directory, states)
        with preserve_cleanup(lambda: files.check(directory, states)):
            reservation = files.deferred(
                _read_file, directory, "reservation.json", states["reservation.json"]
            )
            _reservation(directory, reservation, identity, expected)
            contents = tuple(
                (name, files.deferred(_read_file, directory, name, states[name]))
                for name in PREPARATION_ORDER
            )
            files.check(directory, states)
            yield PreparedStudySnapshot(expected, contents)


def _restore(snapshot, expectations):
    from .retained_study_preparation import restore_study_preparation

    return restore_study_preparation(snapshot, **expectations)


@contextmanager
def _held_restoration(directory, expectations):
    body_error = None
    try:
        digest(expectations["expected_reservation_sha256"])
        digest(expectations["expected_completion_sha256"])
        with _hold_snapshot(
            directory,
            expectations["expected_identity"],
            expectations["expected_reservation_sha256"],
        ) as snapshot:
            restored = _restore(snapshot, expectations)
            try:
                yield restored
            except BaseException as error:
                body_error = error
                raise
    except BaseException as error:
        if not isinstance(error, Exception) or error is body_error:
            raise
        raise StudyPreparationTransportError(
            "invalid_study_preparation_transport"
        ) from None


@contextmanager
def hold_study_preparation(
    directory: Path,
    *,
    expected_identity: dict,
    expected_reservation_sha256: str,
    expected_completion_sha256: str,
    source_spec_bytes: bytes,
    preparation_summary_bytes: bytes,
):
    """Hold immutable retained inputs through the caller's body and final checks."""
    expectations = {
        "expected_identity": expected_identity,
        "expected_reservation_sha256": expected_reservation_sha256,
        "expected_completion_sha256": expected_completion_sha256,
        "source_spec_bytes": source_spec_bytes,
        "preparation_summary_bytes": preparation_summary_bytes,
    }
    with _held_restoration(directory, expectations) as restored:
        yield restored
