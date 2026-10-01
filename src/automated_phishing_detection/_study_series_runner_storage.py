"""Thin held series records over the original create-only publication machinery."""

from contextlib import contextmanager
from functools import partial

from . import _study_preparation_files as files
from . import _study_root_files as storage
from . import execution_receipt as receipt
from ._prepared_failure_context import carry_failure_context


class SeriesRootWriter:
    def __init__(self, held, attempt, public):
        self.held, self.attempt, self.public = held, attempt, public
        self.contents, self.candidate = {}, None
        self.closed = self.failed = False

    def append(self, name, content):
        files.require(not self.closed and not self.failed and not self.held.publishing)
        files.require(
            name
            in {
                "history-import.json",
                "segment-intent.json",
                "segment-accounting.json",
                "series-accounting.json",
                "failure-accounting.json",
            }
        )
        files.require(name not in self.contents and type(content) is bytes)
        try:
            files.deferred(self.held.append, name, content)
            self.contents[name] = content
        except BaseException:
            self.failed = True
            raise

    def complete(self, outputs, public):
        files.require(not self.closed and not self.failed and not self.held.publishing)
        files.deferred(self.held.check)
        self.held.publishing = True
        files.deferred(
            partial(
                receipt.publish_completion,
                self.attempt,
                private_outputs=outputs,
                public_summary=public,
                public_path=self.public,
            )
        )
        content = receipt._json_bytes(public, "series_public")
        added = files.deferred(self.held.read_published, outputs, content)
        reservation = files.deferred(receipt._read_reservation, self.held.root)
        self.candidate = (
            (("attempt/reservation.json", reservation),)
            + tuple((f"attempt/{name}", body) for name, body in self.contents.items())
            + added
        )
        files.deferred(self.held.check)
        return self.candidate


@contextmanager
def hold_root(attempt, public, identity):
    original, writer = None, None
    try:
        with storage.hold(attempt, public, identity) as held:
            writer = SeriesRootWriter(held, attempt, public)
            try:
                yield writer
            except BaseException as error:
                original = error
                raise
    except BaseException as error:
        carry_failure_context(error, original)
        raise
    finally:
        if writer is not None:
            writer.closed = True
