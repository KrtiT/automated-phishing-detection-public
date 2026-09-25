"""Retain a thin same-parent study receipt without scientific or access authority."""

import base64
import json
from contextlib import contextmanager
from functools import partial

from . import _study_preparation_files as files
from . import _study_root_files as storage
from . import _study_root_records as records
from . import execution_receipt as receipt
from ._checkpoint_codec import canonical_bytes
from ._prepared_failure_context import carry_failure_context
from ._study_root_records import StudyRootRetentionError, StudyRootSnapshot

__all__ = ["StudyRootRetentionError", "StudyRootSnapshot", "hold_study_root"]


def _reject(error, original=None):
    carry_failure_context(error, original)
    if not isinstance(error, Exception) or error is original:
        raise error from None
    rejected = StudyRootRetentionError("invalid_study_root_retention")
    carry_failure_context(rejected, error)
    raise rejected from None


class _RootWriter:
    def __init__(self, held, attempt, public, reservation):
        self._held, self._attempt, self._public = held, attempt, public
        self._reservation = reservation
        self._contents, self._confirmed = {}, {}
        self._closed = self._started = self._failed = False
        self._candidate = None

    @property
    def payloads(self):
        return tuple(self._contents.items())

    @property
    def publishing(self):
        return self._held.publishing

    @property
    def candidate(self):
        return self._candidate

    def append(self, name, content):
        try:
            records.require(not self._closed and not self._started and not self._failed)
            records.next_name(self._contents, name)
            records.decode(content)
            self._contents[name] = content
            files.deferred(self._held.append, name, content)
            self._confirmed.update(records.hashes({name: content}))
        except BaseException as error:
            self._failed = True
            _reject(error)

    def _publish(self, outputs, expected):
        files.deferred(self._held.check)
        self._held.publishing = True
        files.deferred(
            partial(
                receipt.publish_completion,
                self._attempt,
                private_outputs=outputs,
                public_summary=json.loads(expected),
                public_path=self._public,
            )
        )
        added = files.deferred(self._held.read_published, outputs, expected)
        original = (("attempt/reservation.json", self._reservation),) + tuple(
            (f"attempt/{name}", content) for name, content in self.payloads
        )
        self._candidate = StudyRootSnapshot(
            self._attempt.reservation_sha256, original + added
        )
        files.deferred(self._held.check)
        return self._candidate

    def complete(self, *, extra_outputs, public_summary):
        try:
            records.require(not self._closed and not self._started and not self._failed)
            self._started = True
            outputs, success = records.private_outputs(self._contents, extra_outputs)
            identity = json.loads(self._reservation)["identity"]
            expected = records.public_bytes(
                self._attempt,
                identity,
                self._contents,
                outputs,
                success,
                public_summary,
            )
            return self._publish(outputs, expected)
        except BaseException as error:
            _reject(error)

    def snapshot(self):
        return canonical_bytes(
            {
                "schema_version": 1,
                "protocol": "study-root-retention-v1",
                "reservation_sha256": self._attempt.reservation_sha256,
                "publishing": self.publishing,
                "confirmed_sha256": dict(self._confirmed),
                "pending_checkpoint_bytes": {
                    name: base64.b64encode(content).decode("ascii")
                    for name, content in self.payloads
                    if name not in self._confirmed
                },
            }
        )


@contextmanager
def hold_study_root(attempt, public_summary, *, expected_identity):
    """Hold before work; candidate acceptance requires all enclosing holders to exit."""
    writer, original = None, None
    try:
        reservation = records.reservation(attempt, expected_identity)
        identity = json.loads(reservation)["identity"]
        with storage.hold(attempt, public_summary, identity) as held:
            writer = _RootWriter(held, attempt, public_summary, reservation)
            try:
                yield writer
            except BaseException as error:
                original = error
                raise
            records.require(writer.candidate is not None)
    except BaseException as error:
        _reject(error, original)
    finally:
        if writer is not None:
            writer._closed = True
