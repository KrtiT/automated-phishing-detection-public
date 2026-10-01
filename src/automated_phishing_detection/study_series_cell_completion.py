"""Retain original held-file lifecycle for a fresh series cell.

Candidates become acceptable only after enclosing holders exit successfully.
This component grants no access and does not observe or launch processes.
"""

import json
from contextlib import contextmanager
from functools import partial

from . import _operational_cell_files as storage
from . import _study_preparation_files as files
from . import execution_receipt as receipt
from . import operational_cell_completion as original
from ._operational_cell_files import OperationalCellCompletionError
from ._operational_process_records import ProcessObservation
from .study_series_cell_acceptance import build_series_cell_public as _build_public
from .study_series_cell_acceptance import (
    verify_series_published_cell as _verify_published,
)
from .study_series_cell_acceptance import verify_series_working_cell as _verify_working

__all__ = ["OperationalCellCompletionError", "hold_series_cell"]


class _Completer(original._Completer):
    def _publish(self):
        public = _build_public(
            self._working, reservation_sha256=self._attempt.reservation_sha256
        )
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

    def _complete(self, expected):
        try:
            storage.require(not self._started and not self._closed)
            self._started = True
            storage.require(type(expected["observation"]) is ProcessObservation)
            payloads = files.deferred(self._held.read_working)
            self._working = _verify_working(
                payloads,
                attempt=self._attempt,
                expected_identity=json.loads(self._identity),
                **expected,
            )
            return self._publish()
        except BaseException as error:
            original._reject(error)

    def complete(
        self,
        *,
        inputs,
        profile_bytes,
        expected_profile_sha256,
        expected_metadata_sha256,
        internal_snapshot,
        external_snapshot,
        observation,
        service_command,
        client_command,
        expected_deadlines,
    ):
        return self._complete(
            dict(
                inputs=inputs,
                profile_bytes=profile_bytes,
                expected_profile_sha256=expected_profile_sha256,
                expected_metadata_sha256=expected_metadata_sha256,
                internal_snapshot=internal_snapshot,
                external_snapshot=external_snapshot,
                observation=observation,
                service_command=service_command,
                client_command=client_command,
                expected_deadlines=expected_deadlines,
            )
        )


@contextmanager
def hold_series_cell(attempt, public_summary, *, expected_identity):
    """Pin before launch; keep failure publication under the observing parent."""
    completer, failure = None, None
    try:
        identity = original._identity(attempt, expected_identity)
        with storage.hold(attempt, public_summary) as held:
            completer = _Completer(held, attempt, public_summary, identity)
            try:
                yield completer
            except BaseException as error:
                failure = error
                raise
            storage.require(completer.candidate is not None)
    except BaseException as error:
        original._reject(error, failure)
    finally:
        if completer is not None:
            completer._closed = True
