"""Fixed thin-root files held across create-only receipt publication."""

import os
import stat
from contextlib import contextmanager

from . import _study_preparation_files as files
from . import execution_receipt as receipt
from ._exception_cleanup import CleanupStack, preserve_cleanup
from ._study_root_records import publication, require


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


class HeldRootFiles:
    def __init__(self, cleanup, root, parent, public_name, attempt, identity):
        self.cleanup, self.root, self.parent = cleanup, root, parent
        self.public_name, self.attempt = public_name, attempt
        self.states = files.authenticate(root, attempt, identity)
        self.pending = self.added = self.evidence = None
        self.publishing = False
        receipt._require_absent(parent, public_name)

    def check(self):
        self.root.check()
        self.parent.check()
        require(stat.S_IMODE(os.fstat(self.root.descriptor).st_mode) == 0o700)
        for name, initial in self.states.items():
            require(capture(self.root, name) == initial)
        if self.added is not None:
            inventory(
                self.root, (*self.states, "finalize.claim", "outcome.json", "evidence")
            )
            inventory(self.evidence, self.output_names)
            for directory, name, unused, initial, mode, content in self.added:
                require(capture(directory, name, mode) == initial)
        elif not self.publishing:
            names = set(self.states)
            if (
                self.pending is not None
                and receipt._entry(self.root, self.pending) is not None
            ):
                names.add(self.pending)
            inventory(self.root, names)
            receipt._require_absent(self.parent, self.public_name)

    def append(self, name, content):
        self.check()
        self.pending = name
        self.states[name] = files.append(self.root, self.states, name, content)
        self.pending = None

    def _capture_published(self, outputs, public):
        self.evidence = files.enter_directory(self.cleanup, self.root.path / "evidence")
        self.output_names = tuple(outputs)
        inventory(
            self.root, (*self.states, "finalize.claim", "outcome.json", "evidence")
        )
        inventory(self.evidence, self.output_names)
        members = (
            (self.root, publication(self.attempt, outputs, public), "attempt", 0o600),
            (self.evidence, outputs, "attempt/evidence", 0o600),
            (self.parent, {self.public_name: public}, "", 0o644),
        )
        self.added = tuple(
            (
                directory,
                name,
                f"{prefix}/{name}" if prefix else "public-summary.json",
                capture(directory, name, mode),
                mode,
                content,
            )
            for directory, entries, prefix, mode in members
            for name, content in entries.items()
        )

    def read_published(self, outputs, public):
        self.check()
        self._capture_published(outputs, public)
        self.check()
        retained = []
        for directory, name, logical, initial, unused, expected in self.added:
            content = read_file(directory, name, initial)
            require(content == expected)
            retained.append((logical, content))
        self.check()
        return tuple(retained)


@contextmanager
def hold(attempt, public, identity):
    path, destination = (
        receipt._absolute_path(attempt.directory),
        receipt._absolute_path(public),
    )
    require(not destination.is_relative_to(path))
    with CleanupStack() as cleanup:
        root = files.enter_directory(cleanup, path)
        parent = files.enter_directory(cleanup, destination.parent)
        held = files.deferred(
            HeldRootFiles, cleanup, root, parent, destination.name, attempt, identity
        )
        with preserve_cleanup(lambda: files.deferred(held.check)):
            yield held
