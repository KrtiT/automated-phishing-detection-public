"""Create-only private physical records with owned, growth-aware log handles."""

import os
import stat
from contextlib import contextmanager
from hashlib import sha256

from . import _operational_input_files as inputs
from . import _study_preparation_files as files
from . import execution_receipt as receipt
from ._checkpoint_codec import canonical_bytes
from ._exception_cleanup import CleanupStack
from ._process_support import _defer_interrupt

LOG_NAMES = ("conditions.jsonl", "stdout.log", "stderr.log")
RECORD_NAMES = ("pre.json", "launch.json", "post.json", "sleep-cleanup.json")
NAMES = RECORD_NAMES + LOG_NAMES


def _metadata(directory, name, stream, initial):
    metadata = os.fstat(stream.fileno())
    files.require(stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1)
    files.require(stat.S_IMODE(metadata.st_mode) == 0o600)
    files.require(receipt._identity(metadata) == receipt._identity(initial))
    located = receipt._entry(directory, name)
    files.require(located is not None)
    files.require(receipt._identity(metadata) == receipt._identity(located))
    files.require(located.st_nlink == 1 and stat.S_IMODE(located.st_mode) == 0o600)
    if name in RECORD_NAMES:
        files.require(files.state(metadata) == files.state(located))
    else:
        files.require(located.st_size >= metadata.st_size)
    return metadata


def _prefix(stream, size, *, hasher=None, start=0):
    hasher = sha256() if hasher is None else hasher
    position = start
    while position < size:
        content = os.pread(stream.fileno(), min(1024 * 1024, size - position), position)
        files.require(bool(content))
        hasher.update(content)
        position += len(content)
    return hasher


def _log(records, name, stream, size):
    previous = records.sizes[name]
    files.require(size >= previous)
    hasher = _prefix(stream, previous)
    files.require(hasher.hexdigest() == records.prefixes[name])
    records.prefixes[name] = _prefix(
        stream, size, hasher=hasher, start=previous
    ).hexdigest()


class SessionRecords:
    def __init__(self, parent, directory, streams):
        self.parent, self.directory, self.streams = parent, directory, streams
        self.initial = {
            name: os.fstat(stream.fileno()) for name, stream in streams.items()
        }
        self.sizes, self.fixed = dict.fromkeys(NAMES, 0), {}
        self.prefixes = dict.fromkeys(LOG_NAMES, sha256().hexdigest())

    def check(self):
        self.parent.check()
        self.directory.check()
        for selected in (self.parent, self.directory):
            files.require(stat.S_IMODE(os.fstat(selected.descriptor).st_mode) == 0o700)
        files.require(set(os.listdir(self.directory.descriptor)) == set(NAMES))
        for name, stream in self.streams.items():
            stream.flush()
            metadata = _metadata(self.directory, name, stream, self.initial[name])
            if name in self.fixed:
                files.require(files.state(metadata) == self.fixed[name])
            elif name in RECORD_NAMES:
                files.require(metadata.st_size == 0)
            else:
                _log(self, name, stream, metadata.st_size)
            self.sizes[name] = metadata.st_size

    def record(self, name, value):
        with _defer_interrupt():
            files.require(name in RECORD_NAMES and name not in self.fixed)
            self.check()
            content, stream = canonical_bytes(value), self.streams[name]
            files.require(stream.write(content) == len(content))
            stream.flush()
            os.fsync(stream.fileno())
            self.fixed[name] = files.state(os.fstat(stream.fileno()))
            self.check()


def _streams(cleanup, directory):
    streams = {}
    for name in NAMES:
        descriptor = files.open_file(
            cleanup,
            directory,
            name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | os.O_NONBLOCK
            | (os.O_APPEND if name in LOG_NAMES else 0),
        )
        os.fchmod(descriptor, 0o600)
        streams[name] = cleanup.enter_context(
            os.fdopen(descriptor, "r+b", buffering=0, closefd=False)
        )
    receipt._sync_directory(directory)
    return streams


def _finish(records):
    records.check()
    for stream in records.streams.values():
        os.fsync(stream.fileno())
    receipt._sync_directory(records.directory)


@contextmanager
def hold_records(path):
    with CleanupStack() as cleanup:
        parent, directory = files.deferred(inputs._create, cleanup, path)
        streams = files.deferred(_streams, cleanup, directory)
        records = SessionRecords(parent, directory, streams)
        files.deferred(records.check)
        cleanup.callback(files.deferred, _finish, records)
        yield records
