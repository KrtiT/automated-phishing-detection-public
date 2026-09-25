"""Retain one preparation attempt without scoring, cleanup or resume authority."""

import base64
from contextlib import contextmanager
from hashlib import sha256

from . import _study_preparation_files as files
from . import execution_receipt as receipt
from ._checkpoint_codec import canonical_bytes
from ._exception_cleanup import CleanupStack
from ._study_preparation_files import StudyPreparationRetentionError

PREPARATION_ORDER = (
    "suffix-rules.dat",
    "group_test.jsonl",
    "source-overlap.json",
    "source-reconstruction.json",
    "publisher-source.json",
    "publisher-summary.json",
    "retained-test.jsonl",
    "quarantine.jsonl",
    "inventory.json",
    "preparation-summary.json",
    "feasibility.json",
    "preparation-complete.json",
)


class _PreparationWriter:
    def __init__(self, directory, attempt, identity):
        self._directory = directory
        self._attempt = attempt
        self._states = files.authenticate(directory, attempt, identity)
        self._contents = {}
        self._confirmed = {}
        self._failed = False
        self._finished = False
        self._closed = False

    def _failure(self, error):
        self._failed = True
        if not isinstance(error, Exception):
            raise error from None
        raise StudyPreparationRetentionError(
            "study_preparation_retention_failed"
        ) from None

    def append(self, name: str, content: bytes) -> None:
        try:
            files.require(not self._failed and not self._finished and not self._closed)
            files.require(type(name) is str and type(content) is bytes)
            position = len(self._contents)
            files.require(position < len(PREPARATION_ORDER))
            files.require(name == PREPARATION_ORDER[position])
            self._contents[name] = content
            self._states[name] = files.append(
                self._directory, self._states, name, content
            )
            self._confirmed[name] = sha256(content).hexdigest()
        except BaseException as error:
            self._failure(error)

    def finish(self) -> tuple[tuple[str, bytes], ...]:
        try:
            files.require(not self._failed and not self._finished and not self._closed)
            files.require(tuple(self._confirmed) == PREPARATION_ORDER)
            files.check(self._directory, self._states)
            self._finished = True
            return tuple(self._contents.items())
        except BaseException as error:
            self._failure(error)

    def _check_finished(self):
        files.require(not self._failed and self._finished)
        files.check(self._directory, self._states)

    def snapshot(self) -> bytes:
        return canonical_bytes(
            {
                "schema_version": 1,
                "protocol_id": "study-preparation-retention-v1",
                "reservation_sha256": self._attempt.reservation_sha256,
                "status": "failed"
                if self._failed
                else "complete"
                if self._finished
                else "pending",
                "confirmed_sha256": dict(self._confirmed),
                "pending_checkpoint_bytes": {
                    name: base64.b64encode(content).decode("ascii")
                    for name, content in self._contents.items()
                    if name not in self._confirmed
                },
            }
        )


@contextmanager
def retain_study_preparation(attempt: receipt.Attempt, *, identity: dict):
    """Hold the attempt through readback and final checks; never delete evidence."""
    writer, body_error = None, None
    try:
        files.require(type(attempt) is receipt.Attempt)
        with CleanupStack() as cleanup:
            directory = files.enter_directory(cleanup, attempt.directory)
            writer = _PreparationWriter(directory, attempt, identity)
            try:
                yield writer
            except BaseException as error:
                body_error = error
                raise
            writer._check_finished()
    except BaseException as error:
        if writer is not None:
            writer._failed = True
        if not isinstance(error, Exception) or error is body_error:
            raise
        raise StudyPreparationRetentionError(
            "study_preparation_retention_failed"
        ) from None
    finally:
        if writer is not None:
            writer._closed = True
