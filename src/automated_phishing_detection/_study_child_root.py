"""Hold the exact fresh root checkpoint prefix before a child's private inputs."""

import os
from contextlib import contextmanager
from hashlib import sha256

from . import _study_preparation_files as files
from ._adopted_study_verification import validate_child_root
from ._exception_cleanup import CleanupStack
from ._prepared_failure_context import carry_failure_context


def _expected(frame):
    result = {
        "reservation.json": frame.root_reservation_sha256,
        "study-intent.json": frame.intent_sha256,
        "prediction-barrier.json": frame.barrier_sha256,
    }
    if frame.role in ("service", "client"):
        result["source-results.json"] = frame.predecessor_sha256
    return result


def _read(cleanup, directory, name, expected):
    initial = files.capture(directory, name)
    descriptor = files.open_file(
        cleanup, directory, name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    )
    files.require(files.state(os.fstat(descriptor)) == initial)
    stream = cleanup.enter_context(os.fdopen(descriptor, "rb", closefd=False))
    content = stream.read()
    files.require(sha256(content).hexdigest() == expected)
    files.require(files.state(os.fstat(descriptor)) == initial)
    files.require(files.capture(directory, name) == initial)
    return content, initial


@contextmanager
def hold_child_root(authorization, frame):
    original = None
    try:
        with CleanupStack() as cleanup:
            directory = files.enter_directory(cleanup, authorization.paths.attempt)
            contents, states = {}, {}
            for name, digest in _expected(frame).items():
                contents[name], states[name] = _read(cleanup, directory, name, digest)
            files.check(directory, states)
            cleanup.callback(files.check, directory, states)
            validate_child_root(authorization, frame, contents)
            try:
                yield contents
            except BaseException as error:
                original = error
                raise
    except BaseException as error:
        carry_failure_context(error, original)
        raise
