"""Prepared-only I/O guards; numerical work and saved reconstruction stay interruptible."""

from contextlib import contextmanager, nullcontext

from . import _external_completion_files as files
from . import source_runner
from ._exception_cleanup import CleanupStack, preserve_cleanup
from ._external_source_checkpoints import ExternalCheckpointWriter
from ._study_preparation_body import deferred_io


def io_scope(preparation):
    return deferred_io() if preparation is not None else nullcontext()


class PreparedExternalCheckpointWriter(ExternalCheckpointWriter):
    def begin(self, provenance):
        with deferred_io():
            return super().begin(provenance)

    def __call__(self, name, content):
        with deferred_io():
            return super().__call__(name, content)

    def complete(self, scientific_outputs):
        with deferred_io():
            return super().complete(scientific_outputs)


def _read_record(directory, name, initial):
    with deferred_io():
        return source_runner._read_file_once(
            directory.path / name, expected_state=initial
        )


def _close(cleanup):
    with deferred_io():
        cleanup.close()


@contextmanager
def snapshot_prepared_external_files(attempt, public_summary):
    cleanup = CleanupStack()
    with preserve_cleanup(lambda: _close(cleanup)):
        with deferred_io():
            attempt, public = files._paths(attempt, public_summary)
            directories = cleanup.enter_context(files._directories(attempt, public))
            files._check_directories(directories)
            captured = files._capture_files(directories)
        snapshot = files.ExternalFileSnapshot(
            tuple(
                (logical, _read_record(directory, name, initial))
                for directory, name, logical, initial in captured
            )
        )
        yield snapshot
        with deferred_io():
            files._recheck_files(directories, captured)
