"""Complete a same-parent cell while original output identities remain held.

A returned candidate is accepted only after every enclosing input/output holder
exits successfully. This helper neither launches children nor grants access.
"""

import json
from contextlib import contextmanager
from functools import partial
from hashlib import sha256

from . import _operational_cell_files as storage
from . import _study_preparation_files as files
from . import execution_receipt as receipt
from ._operational_cell_files import OperationalCellCompletionError
from ._operational_process_records import ProcessObservation
from ._prepared_failure_context import carry_failure_context

__all__ = ["OperationalCellCompletionError", "hold_operational_cell"]


def _verify_working(payloads, **expected):
    from .operational_cell_acceptance import verify_working_cell

    return verify_working_cell(payloads, **expected)


def _build_public(working, reservation):
    from .operational_cell_acceptance import build_cell_public

    return build_cell_public(working, reservation_sha256=reservation)


def _verify_published(payloads, **expected):
    from .operational_cell_acceptance import verify_published_cell

    return verify_published_cell(payloads, **expected)


def _reject(error, original=None):
    carry_failure_context(error, original)
    if not isinstance(error, Exception) or error is original:
        raise error from None
    rejected = OperationalCellCompletionError("invalid_operational_cell_completion")
    carry_failure_context(rejected, error)
    raise rejected from None


def _identity(attempt, expected):
    storage.require(type(attempt) is receipt.Attempt)
    identity = receipt._json_bytes(expected, "operational_cell_identity")
    reservation = receipt._json_bytes(
        {
            "schema_version": 1,
            "status": "reserved",
            "directory": str(receipt._absolute_path(attempt.directory)),
            "identity": json.loads(identity),
        },
        "operational_cell_reservation",
    )
    storage.require(sha256(reservation).hexdigest() == attempt.reservation_sha256)
    return identity


class _Completer:
    def __init__(self, held, attempt, public, identity):
        self._held, self._attempt, self._public = held, attempt, public
        self._identity = identity
        self._started = self._closed = False
        self._working = self._candidate = None

    @property
    def publishing(self):
        return self._held.publishing

    @property
    def working(self):
        return self._working

    @property
    def candidate(self):
        return self._candidate

    def _publish(self):
        public = _build_public(self._working, self._attempt.reservation_sha256)
        expected = receipt._json_bytes(public, "operational_cell_public")
        self._held.publishing = True
        files.deferred(
            partial(
                receipt.publish_completion,
                self._attempt,
                private_outputs=self._working.private_outputs,
                public_summary=public,
                public_path=self._public,
            )
        )
        payloads = files.deferred(self._held.read_published)
        self._candidate = _verify_published(
            payloads, working=self._working, expected_public_bytes=expected
        )
        files.deferred(self._held.check)
        return self._candidate

    def complete(
        self,
        *,
        inputs,
        accepted,
        observation,
        service_command,
        client_command,
        expected_deadlines,
    ):
        try:
            storage.require(not self._started and not self._closed)
            self._started = True
            storage.require(type(observation) is ProcessObservation)
            payloads = files.deferred(self._held.read_working)
            self._working = _verify_working(
                payloads,
                attempt=self._attempt,
                expected_identity=json.loads(self._identity),
                inputs=inputs,
                accepted=accepted,
                observation=observation,
                service_command=service_command,
                client_command=client_command,
                expected_deadlines=expected_deadlines,
            )
            return self._publish()
        except BaseException as error:
            _reject(error)


@contextmanager
def hold_operational_cell(attempt, public_summary, *, expected_identity):
    """Pin before launch; retain partial state and never finalize a failure here."""
    completer, original = None, None
    try:
        identity = _identity(attempt, expected_identity)
        with storage.hold(attempt, public_summary) as held:
            completer = _Completer(held, attempt, public_summary, identity)
            try:
                yield completer
            except BaseException as error:
                original = error
                raise
            storage.require(completer.candidate is not None)
    except BaseException as error:
        _reject(error, original)
    finally:
        if completer is not None:
            completer._closed = True
