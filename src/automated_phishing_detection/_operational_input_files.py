"""Fixed operational input leaves, separate from growing attempt inventories."""

import os
import stat
from contextlib import contextmanager

from . import _study_preparation_files as files
from . import execution_receipt as receipt
from ._exception_cleanup import CleanupStack, preserve_cleanup

ROOT_NAMES = ("accepted-inputs.json",)
CELL_NAMES = ("descriptor.json", "binding.json", "manifest")


def _private(directory):
    files.require(stat.S_IMODE(os.fstat(directory.descriptor).st_mode) == 0o700)
    files.deferred(directory.check)


def _create(cleanup, path):
    parent = files.enter_directory(cleanup, path.parent)
    _private(parent)
    receipt._require_absent(parent, path.name)
    os.mkdir(path.name, 0o700, dir_fd=parent.descriptor)
    initial = receipt._entry(parent, path.name)
    directory = files.enter_directory(cleanup, path)
    files.require(
        receipt._identity(os.fstat(directory.descriptor)) == receipt._identity(initial)
    )
    os.fchmod(directory.descriptor, 0o700)
    receipt._sync_directory(parent)
    return parent, directory


def _check_created(parent, directory, states):
    _private(parent)
    files.check(directory, states)


@contextmanager
def retain(path, payloads):
    with CleanupStack() as cleanup:
        parent, directory = files.deferred(_create, cleanup, path)
        states = {}
        for name, content in payloads:
            states[name] = files.deferred(
                files.append, directory, states, name, content
            )
        files.deferred(_check_created, parent, directory, states)
        with preserve_cleanup(
            lambda: files.deferred(_check_created, parent, directory, states)
        ):
            yield directory.path


def check_all(held):
    for directory, states in held:
        files.check(directory, states)


@contextmanager
def hold(root_path, cell_path):
    with CleanupStack() as cleanup:
        root = files.enter_directory(cleanup, root_path)
        cell = files.enter_directory(cleanup, cell_path)
        files.require(
            receipt._identity(os.fstat(root.descriptor))
            != receipt._identity(os.fstat(cell.descriptor))
        )
        held = tuple(
            (directory, {name: files.capture(directory, name) for name in names})
            for directory, names in ((root, ROOT_NAMES), (cell, CELL_NAMES))
        )
        files.deferred(check_all, held)
        with preserve_cleanup(lambda: files.deferred(check_all, held)):
            yield held


def read_file(directory, name, initial):
    with CleanupStack() as cleanup:
        descriptor = files.open_file(
            cleanup, directory, name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        )
        files.require(files.state(os.fstat(descriptor)) == initial)
        stream = cleanup.enter_context(os.fdopen(descriptor, "rb", closefd=False))
        content = stream.read()
        files.require(len(content) == initial[2])
        files.require(files.state(os.fstat(descriptor)) == initial)
        files.require(files.capture(directory, name) == initial)
        files.deferred(directory.check)
        return content
