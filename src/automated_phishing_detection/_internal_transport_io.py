"""Fixed private transport files with checked states and owned descriptor closure."""

import os
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path

from . import execution_receipt as receipt
from . import source_runner
from ._exception_cleanup import CleanupStack, preserve_cleanup
from ._internal_handoff_validation import require
from ._process_support import _defer_interrupt

NAMES = ("internal-source-handoff.json", "internal-source-overlap.json")


def _directory_state(directory):
    directory.check()
    require(stat.S_IMODE(os.fstat(directory.descriptor).st_mode) == 0o700)
    require(set(os.listdir(directory.descriptor)) == set(NAMES))


def _file_states(directory):
    states = {}
    for name in NAMES:
        metadata = receipt._entry(directory, name)
        require(
            metadata is not None
            and stat.S_ISREG(metadata.st_mode)
            and stat.S_IMODE(metadata.st_mode) == 0o600
            and metadata.st_nlink == 1
        )
        states[name] = source_runner._file_state(metadata)
    return states


def _recheck(directory, states):
    _directory_state(directory)
    require(_file_states(directory) == states)


@contextmanager
def read_snapshot(path):
    with CleanupStack() as cleanup:
        directory = cleanup.enter_context(receipt._directory(path))
        _directory_state(directory)
        states = _file_states(directory)
        with preserve_cleanup(lambda: _recheck(directory, states)):
            contents = {
                name: source_runner._read_file_once(
                    directory.path / name, expected_state=states[name]
                )
                for name in NAMES
            }
            yield contents


class RetainedFiles:
    def __init__(self):
        self.path, self.directory, self.identity = None, None, None
        self.descriptors, self.files = [], {}
        self.complete = False

    def create(self, contents):
        parent = Path(tempfile.gettempdir()).resolve(strict=True)
        with CleanupStack() as assignment:
            assignment.enter_context(_defer_interrupt())
            self.path = Path(tempfile.mkdtemp(prefix="internal-handoff-", dir=parent))
            self.identity = receipt._identity(os.stat(self.path, follow_symlinks=False))
            os.chmod(self.path, 0o700, follow_symlinks=False)
            descriptor = receipt._open_directory(self.path)
            self.descriptors.append(descriptor)
            self.directory = receipt._Directory(self.path, descriptor)
        require(receipt._identity(os.fstat(descriptor)) == self.identity)
        for name, content in zip(NAMES, contents, strict=True):
            self.write(name, content)
        _recheck(self.directory, self.files)
        self.complete = True

    def write(self, name, content):
        with CleanupStack() as assignment:
            assignment.enter_context(_defer_interrupt())
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=self.directory.descriptor,
            )
            self.descriptors.append(descriptor)
            os.fchmod(descriptor, 0o600)
        remaining = memoryview(content)
        while remaining:
            written = os.write(descriptor, remaining)
            require(written > 0)
            remaining = remaining[written:]
        os.fsync(descriptor)
        self.files[name] = source_runner._file_state(os.fstat(descriptor))

    def _check_retained(self):
        if self.complete:
            _recheck(self.directory, self.files)

    def close(self):
        with CleanupStack() as cleanup:
            cleanup.enter_context(_defer_interrupt())
            for descriptor in self.descriptors:
                cleanup.callback(os.close, descriptor)
            self._check_retained()
