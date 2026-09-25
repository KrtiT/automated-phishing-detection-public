"""Hold one attempt across live writes and the fixed17/36 publication transition."""

import os
import stat
from contextlib import contextmanager

from . import _study_preparation_files as files
from . import execution_receipt as receipt
from ._exception_cleanup import CleanupStack, preserve_cleanup
from ._operational_cell_protocol import PRIVATE_NAMES, WORKING_NAMES


class OperationalCellCompletionError(ValueError):
    """Symbolic rejection without private record values or paths."""


def require(condition):
    if not condition:
        raise OperationalCellCompletionError("invalid_operational_cell_completion")


def capture(directory, name, mode=0o600):
    metadata = receipt._entry(directory, name)
    require(metadata is not None and stat.S_ISREG(metadata.st_mode))
    require(metadata.st_nlink == 1 and stat.S_IMODE(metadata.st_mode) == mode)
    return files.state(metadata)


def inventory(directory, names):
    directory.check()
    require(stat.S_IMODE(os.fstat(directory.descriptor).st_mode) == 0o700)
    require(set(os.listdir(directory.descriptor)) == set(names))


def read_file(directory, name, initial):
    with CleanupStack() as cleanup:
        descriptor = files.open_file(
            cleanup, directory, name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        )
        require(files.state(os.fstat(descriptor)) == initial)
        stream = cleanup.enter_context(os.fdopen(descriptor, "rb", closefd=False))
        content = stream.read()
        require(len(content) == initial[2])
        require(files.state(os.fstat(descriptor)) == initial)
        require(files.state(receipt._entry(directory, name)) == initial)
        directory.check()
        return content


class HeldCellFiles:
    def __init__(self, cleanup, root, public_parent, public_name):
        self.cleanup, self.root = cleanup, root
        self.public_parent, self.public_name = public_parent, public_name
        self.original = {"reservation.json": capture(root, "reservation.json")}
        self.working, self.publishing = None, False
        self.evidence, self.added = None, None
        inventory(root, ("reservation.json",))
        receipt._require_absent(public_parent, public_name)

    def check(self):
        self.root.check()
        self.public_parent.check()
        require(stat.S_IMODE(os.fstat(self.root.descriptor).st_mode) == 0o700)
        for name, initial in self.original.items():
            require(capture(self.root, name) == initial)
        if self.added is not None:
            inventory(
                self.root,
                (*WORKING_NAMES, "finalize.claim", "outcome.json", "evidence"),
            )
            inventory(self.evidence, PRIVATE_NAMES)
            for directory, name, unused, initial, mode in self.added:
                require(capture(directory, name, mode) == initial)
        elif self.working is not None and not self.publishing:
            inventory(self.root, WORKING_NAMES)

    def read_working(self):
        require(self.working is None)
        self.check()
        inventory(self.root, WORKING_NAMES)
        captured = {name: capture(self.root, name) for name in WORKING_NAMES}
        require(captured["reservation.json"] == self.original["reservation.json"])
        self.original = captured
        self.working = tuple(
            (name, read_file(self.root, name, initial))
            for name, initial in self.original.items()
        )
        self.check()
        return self.working

    def _capture_published(self):
        require(self.publishing and self.working is not None and self.added is None)
        self.evidence = files.enter_directory(self.cleanup, self.root.path / "evidence")
        inventory(
            self.root, (*WORKING_NAMES, "finalize.claim", "outcome.json", "evidence")
        )
        inventory(self.evidence, PRIVATE_NAMES)
        members = (
            (self.root, ("finalize.claim", "outcome.json"), "attempt", 0o600),
            (self.evidence, PRIVATE_NAMES, "attempt/evidence", 0o600),
            (self.public_parent, (self.public_name,), "", 0o644),
        )
        self.added = tuple(
            (
                directory,
                name,
                f"{prefix}/{name}" if prefix else "public-summary.json",
                capture(directory, name, mode),
                mode,
            )
            for directory, names, prefix, mode in members
            for name in names
        )

    def read_published(self):
        self.check()
        self._capture_published()
        self.check()
        payloads = tuple((f"attempt/{name}", content) for name, content in self.working)
        payloads += tuple(
            (logical, read_file(directory, name, initial))
            for directory, name, logical, initial, unused in self.added
        )
        self.check()
        return payloads


@contextmanager
def hold(attempt, public):
    require(type(attempt) is receipt.Attempt)
    path, destination = (
        receipt._absolute_path(attempt.directory),
        receipt._absolute_path(public),
    )
    require(not destination.is_relative_to(path))
    with CleanupStack() as cleanup:
        root = files.enter_directory(cleanup, path)
        parent = files.enter_directory(cleanup, destination.parent)
        held = files.deferred(HeldCellFiles, cleanup, root, parent, destination.name)
        with preserve_cleanup(lambda: files.deferred(held.check)):
            yield held
