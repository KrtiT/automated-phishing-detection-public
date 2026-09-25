"""Prepared-only ownership guards; numerical work stays outside deferred SIGINT."""

from contextlib import contextmanager, nullcontext
from functools import partial

from . import execution_receipt
from ._exception_cleanup import CleanupStack
from .internal_scientific_checkpoints import ScientificCheckpointWriter


def io_scope(enabled):
    if not enabled:
        return nullcontext()
    from ._study_preparation_body import deferred_io

    return deferred_io()


class PreparedScientificCheckpointWriter(ScientificCheckpointWriter):
    def _write(self, name: str, content: bytes) -> None:
        with io_scope(True):
            super()._write(name, content)


def _exit_directory(exit_method, *arguments):
    with io_scope(True):
        return exit_method(*arguments)


@contextmanager
def _held_directory(path):
    with CleanupStack() as cleanup:
        context = execution_receipt._directory(path)
        cleanup.push(partial(_exit_directory, context.__exit__))
        with io_scope(True):
            directory = context.__enter__()
        yield directory


def completion_directory(path, enabled):
    return _held_directory(path) if enabled else execution_receipt._directory(path)


def completion_read(path, *, expected_state, enabled):
    from .source_runner import _read_file_once

    with io_scope(enabled):
        return _read_file_once(path, expected_state=expected_state)
