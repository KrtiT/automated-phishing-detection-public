"""Held fixed-name attempt writes preserve partial files without staging races."""

import os
import stat
from contextlib import contextmanager

from . import _study_preparation_files as files
from . import execution_receipt as receipt
from ._exception_cleanup import CleanupStack, preserve_cleanup
from ._operational_cell_protocol import WORKING_NAMES

CLIENT_NAMES = (
    "client-role.json",
    "warmup.json",
    "measured.json",
    "run.json",
    "client-failure.json",
)


class OperationalAttemptError(ValueError):
    """Symbolic attempt rejection, without private paths or content."""


def require(condition):
    if not condition:
        raise OperationalAttemptError("invalid_operational_attempt")


def _names(names):
    require(type(names) is tuple and bool(names))
    require(all(type(name) is str for name in names))
    require(len(set(names)) == len(names))
    require(
        set(names)
        <= (set(WORKING_NAMES) - {"reservation.json"}) | {"client-failure.json"}
    )
    require("client-failure.json" not in names or set(names) == set(CLIENT_NAMES))


class AttemptWriter:
    def __init__(self, directory, attempt, names):
        self.directory, self.attempt = directory, attempt
        self.names, self.states = names, {}
        self.states["reservation.json"] = files.capture(directory, "reservation.json")
        receipt._authenticate(attempt, directory)
        self.check()
        require(all(receipt._entry(directory, name) is None for name in self.names))

    def check(self):
        files.deferred(self._check)

    def _check(self):
        self.directory.check()
        require(stat.S_IMODE(os.fstat(self.directory.descriptor).st_mode) == 0o700)
        names = set(os.listdir(self.directory.descriptor))
        require(names <= set(WORKING_NAMES) | {"client-failure.json"})
        for name in names:
            files.capture(self.directory, name)
        for name, initial in self.states.items():
            require(files.capture(self.directory, name) == initial)
        self.directory.check()

    def _append(self, name, content):
        require(name in self.names and type(content) is bytes)
        self.check()
        receipt._require_absent(self.directory, name)
        initial = files.write_file(self.directory, name, content, 0o600)
        receipt._sync_directory(self.directory)
        files.readback(self.directory, name, content, initial)
        self.states[name] = initial
        self.check()

    def retain(self, name, content):
        files.deferred(self._append, name, content)


@contextmanager
def held_attempt_writer(attempt, *, names):
    _names(names)
    require(type(attempt) is receipt.Attempt)
    with CleanupStack() as cleanup:
        directory = files.enter_directory(cleanup, attempt.directory)
        writer = files.deferred(AttemptWriter, directory, attempt, names)
        with preserve_cleanup(lambda: files.deferred(writer.check)):
            yield writer
