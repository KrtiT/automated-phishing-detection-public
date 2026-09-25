"""Held-descriptor checks for private, create-only preparation retention."""

import os
import stat
from functools import partial
from hashlib import sha256

from . import execution_receipt as receipt
from ._exception_cleanup import CleanupStack
from ._process_support import _defer_interrupt


class StudyPreparationRetentionError(ValueError):
    """Symbolic rejection without private paths, content or exception details."""


def require(condition):
    if not condition:
        raise StudyPreparationRetentionError("invalid_study_preparation_retention")


def deferred(function, *arguments):
    with CleanupStack() as guard:
        guard.enter_context(_defer_interrupt())
        return function(*arguments)


def enter_directory(cleanup, path):
    with CleanupStack() as assignment:
        assignment.enter_context(_defer_interrupt())
        manager = receipt._directory(path)
        directory = manager.__enter__()
        cleanup.push(partial(deferred, manager.__exit__))
    return directory


def open_file(cleanup, directory, name, flags, mode=0o600):
    with CleanupStack() as assignment:
        assignment.enter_context(_defer_interrupt())
        descriptor = os.open(name, flags, mode, dir_fd=directory.descriptor)
        cleanup.callback(deferred, os.close, descriptor)
    return descriptor


def state(metadata):
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_nlink,
        metadata.st_mode,
    )


def capture(directory, name):
    metadata = receipt._entry(directory, name)
    require(
        metadata is not None
        and stat.S_ISREG(metadata.st_mode)
        and metadata.st_nlink == 1
        and stat.S_IMODE(metadata.st_mode) == 0o600
    )
    return state(metadata)


def check(directory, states):
    deferred(directory.check)
    require(stat.S_IMODE(os.fstat(directory.descriptor).st_mode) == 0o700)
    require(set(os.listdir(directory.descriptor)) == set(states))
    for name, initial in states.items():
        require(capture(directory, name) == initial)
    deferred(directory.check)


def authenticate(directory, attempt, identity):
    require(type(attempt) is receipt.Attempt and type(identity) is dict)
    initial = capture(directory, "reservation.json")
    check(directory, {"reservation.json": initial})
    content = deferred(receipt._read_reservation, directory)
    expected = receipt._json_bytes(
        {
            "schema_version": 1,
            "status": "reserved",
            "directory": str(directory.path),
            "identity": identity,
        },
        "reservation",
    )
    require(content == expected)
    require(sha256(content).hexdigest() == attempt.reservation_sha256)
    check(directory, {"reservation.json": initial})
    return {"reservation.json": initial}


def readback(directory, name, expected, initial):
    with CleanupStack() as cleanup:
        descriptor = open_file(
            cleanup,
            directory,
            name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
        require(state(os.fstat(descriptor)) == initial)
        stream = cleanup.enter_context(os.fdopen(descriptor, "rb", closefd=False))
        content = stream.read()
        require(content == expected)
        require(sha256(content).digest() == sha256(expected).digest())
        require(state(os.fstat(descriptor)) == initial)
        require(capture(directory, name) == initial)
        deferred(directory.check)


def write_file(directory, name, content, mode):
    with CleanupStack() as cleanup:
        descriptor = open_file(
            cleanup,
            directory,
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            mode,
        )
        stream = cleanup.enter_context(os.fdopen(descriptor, "wb", closefd=False))
        os.fchmod(descriptor, mode)
        stream.write(content)
        stream.flush()
        receipt.transformer_pipeline._sync_descriptor(descriptor)
        initial = state(os.fstat(descriptor))
        require(capture(directory, name) == initial)
        deferred(directory.check)
        return initial


def append(directory, states, name, content):
    check(directory, states)
    deferred(receipt._require_absent, directory, name)
    initial = write_file(directory, name, content, 0o600)
    deferred(receipt._sync_directory, directory)
    require(capture(directory, name) == initial)
    readback(directory, name, content, initial)
    check(directory, {**states, name: initial})
    return initial
